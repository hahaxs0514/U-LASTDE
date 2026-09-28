import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .config import Config


# ====================================================================================
#                              工具函数
# ====================================================================================

def _adjust_time_1d(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    调整 1D 张量的时间维度（末端裁剪或复制填充）
    Args:
        x: (B*, C, T)
        target_T: 目标时间步数
    Returns:
        (B*, C, target_T)
    """
    T = x.shape[-1]
    if T == target_T:
        return x
    elif T > target_T:
        return x[..., :target_T]
    else:
        pad = target_T - T
        return F.pad(x, (0, pad), mode="replicate")


def _adjust_time_4d(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    调整 4D 张量的时间维度（末端裁剪或复制填充）
    Args:
        x: (B, T, N, C)
        target_T: 目标时间步数
    Returns:
        (B, target_T, N, C)
    """
    T = x.shape[1]
    if T == target_T:
        return x
    if T > target_T:
        return x[:, :target_T]
    
    # 填充时需要转换维度以使用 replicate 模式
    pad = target_T - T
    x = x.permute(0, 2, 3, 1)  # (B, N, C, T)
    x = F.pad(x, (0, pad), mode='replicate')
    x = x.permute(0, 3, 1, 2)  # (B, target_T, N, C)
    return x


def row_normalize(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    行归一化（Random-Walk 归一化）
    Args:
        A: (B, N, N) 邻接矩阵（已加自环，非负）
        eps: 平滑项防止除零
    Returns:
        (B, N, N) 每行和为1的随机游走矩阵
    """
    deg = A.sum(dim=-1, keepdim=True) + eps
    return A / deg


def get_activation(activation: str) -> nn.Module:
    """根据配置字符串返回激活函数"""
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu' or 'relu'.")


# ====================================================================================
#                         车道融合模块（2种方式）
# ====================================================================================

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    对被 mask 的位置置为 -inf 后做 softmax；全无效时返回全 0
    Args:
        scores: (..., L) 注意力分数
        mask:   (..., L) 1=有效, 0=无效
        dim:    softmax 维度
        eps:    数值稳定项
    Returns:
        (..., L) 归一化的注意力权重
    """
    neg_large = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(mask == 0, neg_large)
    attn = torch.softmax(masked_scores, dim=dim)

    # 再次乘 mask 并显式归一，避免全 mask 产生 NaN
    attn = attn * mask.to(dtype=attn.dtype)
    attn_sum = attn.sum(dim=dim, keepdim=True).clamp_min(eps)
    attn = attn / attn_sum
    return attn


class AttentionLaneFusion(nn.Module):
    """
    方式1: 时序CNN + 车道自注意力 + CLS池化
    先做时序CNN聚合(沿T)，再做车道自注意力(可多层)+CLS池化
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 时序 CNN: 在时间维 T 上做 1D 卷积
        cnn_layers = []
        in_channels = config.n_features
        for out_channels in config.lane_cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=config.lane_cnn_kernel,
                          padding=config.lane_cnn_kernel // 2),
                nn.BatchNorm1d(out_channels),
                get_activation(config.activation),
                nn.Dropout(config.lane_dropout)
            ])
            in_channels = out_channels
        self.temporal_cnn = nn.Sequential(*cnn_layers)
        D = config.lane_cnn_channels[-1]

        # CLS Token（用于车道级池化）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))

        # 车道自注意力块（可多层）
        self.lane_layers = nn.ModuleList([
            LaneAttentionBlock(
                embed_dim=D,
                num_heads=config.lane_num_heads,
                dropout=config.lane_dropout,
                activation=config.activation
            ) for _ in range(config.lane_num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, N, L, F)
            mask: (B, T, N, L)   1=有效, 0=无效
        Returns:
            out:  (B, T, N, D)
        """
        B, T, N, L, F = x.shape
        D = self.config.lane_cnn_channels[-1]

        # === 阶段一: 沿时间维 T 做 CNN（保持 T 不变）===
        # (B, T, N, L, F) → (B*N*L, F, T)
        x_time = x.permute(0, 2, 3, 4, 1).reshape(B * N * L, F, T)
        x_time = self.temporal_cnn(x_time)  # (B*N*L, D, T)
        # 回到 (B, T, N, L, D)
        x_emb = x_time.permute(0, 2, 1).reshape(B, N, L, T, D).permute(0, 3, 1, 2, 4)

        # === 阶段二: 车道间自注意力 + CLS 池化 ===
        # 将 (B, T, N) 展平为 batch 维
        seq = x_emb.reshape(B * T * N, L, D)  # (B*T*N, L, D)

        # 拼接 CLS（放在最前）
        cls = self.cls_token.expand(seq.shape[0], 1, D)  # (B*T*N, 1, D)
        seq = torch.cat([cls, seq], dim=1)               # (B*T*N, L+1, D)

        # key_padding_mask：True=忽略；CLS 永远 False
        kpm_cls = torch.zeros((B, T, N, 1), dtype=torch.bool, device=x.device)
        kpm_lanes = (mask == 0)                          # (B,T,N,L)
        key_padding_mask = torch.cat([kpm_cls, kpm_lanes], dim=-1)  # (B,T,N,L+1)
        key_padding_mask = key_padding_mask.reshape(B * T * N, L + 1)

        # 多层自注意力
        for block in self.lane_layers:
            seq = block(seq, key_padding_mask=key_padding_mask)  # (B*T*N, L+1, D)

        # 取 CLS 输出作为融合后的路段-时刻表示
        out = seq[:, 0, :]              # (B*T*N, D)
        out = out.reshape(B, T, N, D)   # (B, T, N, D)

        return out


class LaneAttentionBlock(nn.Module):
    """
    单层车道自注意力块（Pre-Norm）:
    seq -> LN -> MHA -> Dropout -> Residual -> LN -> FFN -> Dropout -> Residual
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, seq: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq: (B_seq, S=L+1, D)
            key_padding_mask: (B_seq, S)  True=忽略, False=保留
        Returns:
            (B_seq, S, D)
        """
        # 自注意力（Pre-Norm）
        qkv = self.norm1(seq)
        attn_out, _ = self.attn(qkv, qkv, qkv,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        seq = seq + self.attn_drop(attn_out)

        # FFN（Pre-Norm）
        seq = seq + self.ffn(self.norm2(seq))
        return seq


class DeepSetsLaneFusion(nn.Module):
    """
    方式2: Deep Sets（phi -> masked pooling -> rho）
    使用时序CNN前处理，统一使用Pre-Norm架构
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pool_type = config.deepsets_pool

        # 时序 CNN（Pre-Norm 架构）
        cnn_layers = []
        in_channels = config.n_features
        for out_channels in config.lane_cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=config.lane_cnn_kernel,
                          padding=config.lane_cnn_kernel // 2),
                nn.BatchNorm1d(out_channels),
                get_activation(config.activation),
                nn.Dropout(config.lane_dropout)
            ])
            in_channels = out_channels
        self.temporal_cnn = nn.Sequential(*cnn_layers)
        D = config.lane_cnn_channels[-1]

        # DeepSets 的 phi 和 rho（Pre-Norm 架构）
        hidden = config.deepsets_hidden
        self.phi = self._build_mlp(D, hidden, config.lane_cnn_channels[-1], config.lane_dropout, config.activation)
        self.rho = self._build_mlp(config.lane_cnn_channels[-1], hidden, config.lane_cnn_channels[-1], config.lane_dropout, config.activation)

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, 
                   dropout: float, activation: str) -> nn.Sequential:
        """构建 MLP（Pre-Norm 架构）"""
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def _masked_pool(self, x, mask, dim):
        """带 mask 的池化"""
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        if self.pool_type == "sum":
            return (x * m).sum(dim=dim)
        else:  # mean
            wsum = m.sum(dim=dim).clamp_min(1e-6)
            return (x * m).sum(dim=dim) / wsum

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, N, L, F)
            mask: (B, T, N, L)
        Returns:
            out:  (B, T, N, D)
        """
        B, T, N, L, F = x.shape

        # 时序 CNN
        x_time = x.permute(0, 2, 3, 4, 1).reshape(B * N * L, F, T)
        x_time = self.temporal_cnn(x_time)
        D = x_time.shape[1]
        x_emb = x_time.permute(0, 2, 1).reshape(B, N, L, T, D).permute(0, 3, 1, 2, 4)

        # DeepSets: phi -> pooling -> rho
        h = self.phi(x_emb)  # (B, T, N, L, D)
        pooled = self._masked_pool(h, mask, dim=3)  # (B, T, N, D)
        out = self.rho(pooled)  # (B, T, N, D)

        return out


# ====================================================================================
#                         统一入口模块
# ====================================================================================

class LaneFusionModule(nn.Module):
    """
    统一的车道融合模块入口
    根据 config.lane_fusion_type 选择具体实现
    
    支持两种方式:
    1. "attention": 时序CNN + 车道自注意力 + CLS池化
    2. "deepsets": Deep Sets（phi -> pooling -> rho）
    
    所有方式统一使用:
    - 激活函数: config.activation ('gelu' or 'relu')
    - 归一化: LayerNorm
    - 架构: Pre-Norm
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.abn_idx = int(config.abnormal_day_index)
        self.use_all_days = config.use_all_days

        if self.use_all_days:
            self.fusion = MultiDayLaneFusion(config)
        else:
            # 保持原单天行为
            if config.lane_fusion_type == "attention":
                self.fusion = AttentionLaneFusion(config)
            elif config.lane_fusion_type == "deepsets":
                self.fusion = DeepSetsLaneFusion(config)
            else:
                raise ValueError(f"Unknown lane_fusion_type: {config.lane_fusion_type}")

    @staticmethod
    def _ndim(x: torch.Tensor) -> int:
        return x.dim()

    def _slice_abnormal_day(self, x: torch.Tensor, mask: torch.Tensor):
        """
        将 6D 输入切成单天 5D；若本就 5D 则原样返回。
        期望：
          x:    (B, Dd, T, N, L, F) 或 (B, T, N, L, F)
          mask: (B, Dd, T, N, L)     或 (B, T, N, L)
        返回：
          x_day:    (B, T, N, L, F)
          mask_day: (B, T, N, L)
        """
        if self._ndim(x) == 6:
            B, Dd, T, N, L, F = x.shape
            if not (0 <= self.abn_idx < Dd):
                raise IndexError(f"abnormal_day_index={self.abn_idx} out of range for n_days={Dd}")
            x_day   = x[:, self.abn_idx]      # (B, T, N, L, F)
            mask_day= mask[:, self.abn_idx]   # (B, T, N, L)
            return x_day, mask_day
        elif self._ndim(x) == 5:
            # 调用方已经只传了单天
            return x, mask
        else:
            raise ValueError(f"LaneFusionModule expects 5D or 6D input, got {self._ndim(x)}D")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        - use_all_days=True:
            x:    (B, Dd, T, N, L, F)
            mask: (B, Dd, T, N, L)
            -> (abn, norm): ((B, T, N, D), (B, T, N, D))
        - use_all_days=False:
            x:    (B, T, N, L, F) 或 (B, Dd, T, N, L, F)
            mask: (B, T, N, L)   或 (B, Dd, T, N, L)
            -> (abn, None): ((B, T, N, D), None)
        """
        if self.use_all_days:
            if self._ndim(x) != 6 or self._ndim(mask) != 5:
                raise ValueError("use_all_days=True requires x:(B,Dd,T,N,L,F) and mask:(B,Dd,T,N,L)")
            return self.fusion(x, mask)
        else:
            x_day, mask_day = self._slice_abnormal_day(x, mask)  # 统一成 5D
            out = self.fusion(x_day, mask_day)
            return out, None


# ====================================================================================
#                         多天融合模块
# ====================================================================================

class MultiDayLaneFusion(nn.Module):
    """
    先逐天做车道融合，再简单聚合为 abn 和 norm
    - abn: 直接取第一天（abnormal_day_index=0）
    - norm: 对其他参考天做平均
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 逐天车道融合器
        if config.lane_fusion_type == "attention":
            self.per_day_fuser = AttentionLaneFusion(config)
        elif config.lane_fusion_type == "deepsets":
            self.per_day_fuser = DeepSetsLaneFusion(config)
        else:
            raise ValueError(f"Unknown lane_fusion_type: {config.lane_fusion_type}")

    def forward(self, x, mask):
        """
        x:    (B, Dd, T, N, L, F)
        mask: (B, Dd, T, N, L)
        返回: (abn, norm)，两者形状都为 (B, T, N, D)
        """
        B, Dd, T, N, L, F = x.shape
        abn_idx = self.config.abnormal_day_index

        # 逐天融合
        h_list = []
        for d in range(Dd):
            x_d = x[:, d]              # (B, T, N, L, F)
            m_d = mask[:, d]           # (B, T, N, L)
            h_d = self.per_day_fuser(x_d, m_d)  # (B, T, N, D)
            h_list.append(h_d.unsqueeze(1))     # (B, 1, T, N, D)
        H_days = torch.cat(h_list, dim=1)       # (B, Dd, T, N, D)

        # 1) 目标天：直接取第一天
        abn = H_days[:, abn_idx]  # (B, T, N, D)

        # 2) 参考天：对其他天做平均
        if Dd == 1:
            norm = torch.zeros_like(abn)
        else:
            # 创建参考天的mask（排除目标天）
            ref_indices = [d for d in range(Dd) if d != abn_idx]
            ref_days = H_days[:, ref_indices]  # (B, Dd-1, T, N, D)
            norm = ref_days.mean(dim=1)  # (B, T, N, D)

        return abn, norm


# ====================================================================================
#                         残差差异融合模块
# ====================================================================================

class ResidualDiffFusion(nn.Module):
    """
    Diff-SE 融合版本
    - 若 b 存在：对 diff = a - b 计算 SE 模块，返回 SE(diff)
    - 若 b 为 None：对 a 计算 SE 模块，返回 SE(a)
    """
    def __init__(self, C, dropout=0.1, activation='gelu', tau: float = 1.0):
        super().__init__()
        assert tau > 0, "tau must be positive"
        self.tau = tau
        self.C = C

        # SE 模块的缩减比例
        reduction = 16
        hidden_dim = max(C // reduction, 8)

        # SE 模块
        self.se = nn.Sequential(
            nn.Linear(C, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, C),
            nn.Sigmoid()
        )

    def forward(self, a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        a: (B,T,N,C)  待检测嵌入
        b: (B,T,N,C)  正常对照嵌入（同周期），可选
        返回: (B,T,N,C)
        """
        if b is not None:
            # 使用 diff
            diff = a - b  # (B,T,N,C)
            
            # SE for diff: 在时间维度上全局平均池化
            diff_pooled = diff.mean(dim=1)  # (B,N,C)
            diff_weight = self.se(diff_pooled)  # (B,N,C)
            diff_weighted = diff * diff_weight.unsqueeze(1)  # (B,T,N,C)
            
            return diff_weighted
        else:
            # 使用 a
            a_pooled = a.mean(dim=1)  # (B,N,C)
            a_weight = self.se(a_pooled)  # (B,N,C)
            a_weighted = a * a_weight.unsqueeze(1)  # (B,T,N,C)
            
            return a_weighted


# ====================================================================================
#                         时域卷积模块
# ====================================================================================

class SE1d(nn.Module):
    """
    标准 SE：沿时间维做 GAP，产生通道权重后对每个时间步广播
    输入:  (B*, C, T)  输出: (B*, C, T)
    """
    def __init__(self, channels: int, reduction: int = 4, act: str = "relu"):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)          # (B*, C, 1)
        self.fc1  = nn.Conv1d(channels, mid, 1, bias=True)
        self.act  = get_activation(act)
        self.fc2  = nn.Conv1d(mid, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)               # (B*, C, 1)
        s = self.fc2(self.act(self.fc1(s)))  # (B*, C, 1)
        w = self.gate(s)               # (B*, C, 1)
        return x * w                   # 按通道缩放, 对时间维广播


class TemporalConvGLU(nn.Module):
    """
    三种时序建模实现（仅在时间维做卷积，统一 keep_time=True）：
      - 'glu'       : Conv1d(2*C_out) -> 拆分 (A,G) 做 GLU + 残差
      - 'conv'  : Conv1d（中间激活
      - 'inception' : 多核并联，使用SE模块融合

    输入:  (B, T, N, C_in)
    输出:  (B, T, N, C_out)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 3,
                 dropout: float = 0.1, config=None):
        super().__init__()
        assert config is not None, "config 不能为空"
        assert k % 2 == 1, "建议奇数核以便 SAME 对齐"

        self.k = k
        self.c_in = c_in
        self.c_out = c_out
        self.se_reduction = 2

        # 配置
        self.variant = config.temporal_variant
        self.act_name = config.activation

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 统一使用 SAME padding
        pad = k // 2

        # ====== 分支构建 ======
        if self.variant == "glu":
            # GLU 分支
            self.conv_glu = nn.Conv1d(c_in, 2 * c_out, kernel_size=k, padding=pad, bias=True)
            self.proj_glu = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

        elif self.variant == "conv":
            # Two-Conv 分支（无残差）
            self.act1 = get_activation(self.act_name)
            self.act2 = get_activation(self.act_name)

            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=True)
            self.norm1 = nn.BatchNorm1d(c_out)

            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, bias=True)
            self.norm2 = nn.BatchNorm1d(c_out)

        elif self.variant == "inception":
            # 多核并联
            self.kernel_set = config.temporal_kernel_set
            assert len(self.kernel_set) >= 1, "temporal_kernel_set 至少包含一个核尺寸"

            # 通道分配：各分支拼接后总通道 = c_out
            M = len(self.kernel_set)
            base = c_out // M
            self.branch_out = [base] * M
            self.branch_out[-1] += (c_out - sum(self.branch_out))  # 修正到恰好 c_out

            # filter 分支
            self.filter_convs = nn.ModuleList()

            for i, ksz in enumerate(self.kernel_set):
                assert ksz >= 1
                pad_i = ksz // 2
                c_br  = self.branch_out[i]
                norm = nn.BatchNorm1d(c_br)
                act  = get_activation(self.act_name)
                self.filter_convs.append(nn.Sequential(
                    nn.Conv1d(c_in, c_br, kernel_size=ksz, padding=pad_i, bias=True),
                    norm,
                    act,
                ))

            self.inception_se = SE1d(c_out, reduction=self.se_reduction, act=self.act_name)
        else:
            raise ValueError(f"Unknown temporal_variant: {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C_in)
        Returns:
            out: (B, T, N, C_out)
        """
        B, T, N, Cin = x.shape
        # (B, T, N, C) -> (B*N, C, T)
        x1 = x.permute(0, 2, 3, 1).reshape(B * N, Cin, T)

        if self.variant == "glu":
            # ===== GLU =====
            y = self.conv_glu(x1)                 # (BN, 2*C_out, T)
            A, G = torch.chunk(y, 2, dim=1)       # (BN, C_out, T)

            res = self.proj_glu(x1)               # (BN, C_out, T)

            out = (A + res) * torch.sigmoid(G)    # GLU
            out = self.dropout(out)               # (BN, C_out, T)

        elif self.variant == "conv":
            # ===== Two-Conv（无残差）=====
            y1 = self.conv1(x1)                   # (BN, C_out, T)
            y1 = self.norm1(y1)
            y1 = self.act1(y1)

            y2 = self.conv2(y1)                   # (BN, C_out, T)
            y2 = self.norm2(y2)
            y2 = self.act2(y2)

            out = self.dropout(y2)                # (BN, C_out, T)

        else:
            # ===== Inception（多核）=====
            f_list = [conv(x1) for conv in self.filter_convs]   # 各分支 (BN, c_br, T)
            F_cat = torch.cat(f_list, dim=1)         # (BN, C_out, T)
            out   = self.inception_se(F_cat)         # SE 通道融合
            out   = self.dropout(out)                # (BN, C_out, T)
            
        # 还原: (BN, C_out, T) -> (B, T, N, C_out)
        T_out = out.shape[-1]
        out = out.reshape(B, N, self.c_out, T_out).permute(0, 3, 1, 2)
        return out


# ====================================================================================
#                         空间图卷积模块
# ====================================================================================

# class _DiffusionGConvLayer(nn.Module):
#     """
#     单层扩散图卷积（支持有向前/后两路）
    
#     spatial_K 语义：
#     - K=0: 只聚焦自身信息（0跳）
#     - K=1: 聚合 0跳（自身）+ 1跳邻居
#     - K=2: 聚合 0跳 + 1跳 + 2跳
    
#     实现：sum_{k=0}^{K} [ (S_f^k) X W_k^f + (S_b^k) X W_k^b ]
#     其中 S^0 = I（恒等矩阵）
#     """
#     def __init__(self, c_in: int, c_out: int, K: int = 3, 
#                  separate_dirs: bool = True, 
#                  dropout: float = 0.0, activation: str = 'gelu'):
#         super().__init__()
#         assert K >= 0, "K must be >= 0"
#         self.K = K
#         self.sep = separate_dirs
        
#         # 添加 Pre-Norm
#         self.norm = nn.LayerNorm(c_in)
        
#         # K+1 个权重矩阵（对应 0 到 K 跳）
#         num_hops = K + 1
#         if self.sep:
#             self.theta_f = nn.ModuleList([nn.Linear(c_in, c_out, bias=False) for _ in range(num_hops)])
#             self.theta_b = nn.ModuleList([nn.Linear(c_in, c_out, bias=False) for _ in range(num_hops)])
#         else:
#             self.theta = nn.ModuleList([nn.Linear(2 * c_in, c_out, bias=False) for _ in range(num_hops)])

#         # 使用偏置
#         self.bias = nn.Parameter(torch.zeros(c_out))
        
#         # 激活函数
#         self.act = get_activation(activation)
        
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, W_f: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x:   (B, T, N, C_in)
#             W_f: (B, N, N) 前向邻接
#             W_b: (B, N, N) 后向邻接
#         Returns:
#             (B, T, N, C_out)
#         """
#         B, T, N, Cin = x.shape

#         # Pre-Norm
#         x = self.norm(x)

#         # 合并时间维度
#         X = x.reshape(B * T, N, Cin)
#         Sf = row_normalize(W_f).repeat_interleave(T, dim=0)
#         Sb = row_normalize(W_b).repeat_interleave(T, dim=0)

#         # k=0: 恒等映射（0跳，自身信息）
#         Xk_f = X
#         Xk_b = X

#         # 聚合从 0 到 K 跳
#         if self.sep:
#             out = self.theta_f[0](Xk_f) + self.theta_b[0](Xk_b)
#             for k in range(1, self.K + 1):  # k = 1, 2, ..., K
#                 Xk_f = torch.bmm(Sf, Xk_f)  # k跳邻居
#                 Xk_b = torch.bmm(Sb, Xk_b)
#                 out += self.theta_f[k](Xk_f) + self.theta_b[k](Xk_b)
#         else:
#             out = self.theta[0](torch.cat([Xk_f, Xk_b], dim=-1))
#             for k in range(1, self.K + 1):
#                 Xk_f = torch.bmm(Sf, Xk_f)
#                 Xk_b = torch.bmm(Sb, Xk_b)
#                 out += self.theta[k](torch.cat([Xk_f, Xk_b], dim=-1))

#         # 添加偏置
#         out = out + self.bias

#         # 激活 + Dropout
#         out = self.act(out)
#         out = self.dropout(out)
        
#         return out.reshape(B, T, N, -1)

class _DiffusionGConvLayer(nn.Module):
    """
    单层无向扩散图卷积（仅用无向邻接，不分前/后向）
    """
    def __init__(self, c_in: int, c_out: int, K: int = 3, 
                 dropout: float = 0.0, activation: str = 'gelu'):
        super().__init__()
        assert K >= 0, "K must be >= 0"
        self.K = K

        # 添加 Pre-Norm
        self.norm = nn.LayerNorm(c_in)
        
        # K+1 个权重矩阵（对应 0 到 K 跳）
        num_hops = K + 1
        self.theta = nn.ModuleList([nn.Linear(c_in, c_out, bias=False) for _ in range(num_hops)])

        # 使用偏置
        self.bias = nn.Parameter(torch.zeros(c_out))
        
        # 激活函数
        self.act = get_activation(activation)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, W_f: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, C_in)
            W_f: (B, N, N) 前向邻接
            W_b: (B, N, N) 后向邻接
        Returns:
            (B, T, N, C_out)
        """
        B, T, N, Cin = x.shape

        # Pre-Norm
        x = self.norm(x)

        # 合并时间维度
        X = x.reshape(B * T, N, Cin)

        # 合并前后向邻接为无向邻接
        W_undir = W_f + W_b           # (B, N, N)
        S = row_normalize(W_undir)    # (B, N, N)
        S = S.repeat_interleave(T, dim=0)     # (B*T, N, N)

        # k=0: 恒等映射（0跳，自身信息）
        Xk = X

        # 聚合从 0 到 K 跳
        out = self.theta[0](Xk)
        for k in range(1, self.K + 1):
            Xk = torch.bmm(S, Xk)          # k跳邻居
            out += self.theta[k](Xk)

        # 添加偏置
        out = out + self.bias

        # 激活 + Dropout
        out = self.act(out)
        out = self.dropout(out)
        
        return out.reshape(B, T, N, -1)

class _AttnKHopBiasLayer(nn.Module):
    """
    单层空间注意力（多跳扩散先验 → 一次 MHA）
    
    spatial_K 语义：
    - K=0: 只聚焦自身信息（0跳）
    - K=1: 聚合 0跳 + 1跳邻居
    - K=2: 聚合 0跳 + 1跳 + 2跳
    """
    def __init__(self, dim: int, K: int = 3, num_heads: int = 4, 
                 dropout: float = 0.0, clamp_val: float = 5.0, 
                 eps: float = 1e-6, activation: str = 'gelu'):
        super().__init__()
        assert K >= 0, "K must be >= 0"
        self.dim = dim
        self.K = K
        self.h = num_heads
        self.eps = eps
        self.clamp_val = float(clamp_val)

        # 两个方向的 MHA
        self.attn_fw = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                             dropout=dropout, batch_first=True)
        self.attn_bw = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                             dropout=dropout, batch_first=True)

        # K+1 维 *每个头* 的凸权参数: (h, K+1)
        num_hops = K + 1
        self.alpha_fw = nn.Parameter(torch.zeros(num_heads, num_hops))  # (h, K+1)
        self.alpha_bw = nn.Parameter(torch.zeros(num_heads, num_hops))  # (h, K+1)
        
        # 方向融合
        self.lin_sym  = nn.Linear(dim, dim)
        self.lin_anti = nn.Linear(dim, dim)
        self.dir_gate = nn.Parameter(torch.tensor(0.0))

        # 合并与残差
        self.merge_dropout = nn.Dropout(dropout)

        # Pre-Norm & FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_multihop_mixture(S: torch.Tensor, K: int, alphas: torch.Tensor,
                                eps: float, clamp_val: float) -> torch.Tensor:
        """
        构建“每个 head 一套”的多跳混合 additive bias

        Args:
            S:       (B*T, N, N) 行归一化随机游走矩阵
            K:       最大跳数
            alphas:  (H, K+1) 每个 head 一组 multi-hop 权重参数
            eps:     数值稳定项
            clamp_val: 裁剪值

        Returns:
            bias: (B*T * H, N, N)，已经展开好 head 维度，可以直接给 MHA 当 attn_mask
        """
        BT, N, _ = S.shape
        H, K1 = alphas.shape
        assert K1 == K + 1, f"alphas.shape[-1]={K1} must be K+1={K+1}"

        # (H, K+1) -> 每个 head 一条权重
        ws = torch.softmax(alphas, dim=-1)  # (H, K+1)

        # P^0 = I，broadcast 到 (BT, H, N, N)
        P0 = torch.eye(N, device=S.device, dtype=S.dtype)         # (N, N)
        P  = P0.view(1, 1, N, N).expand(BT, H, N, N)              # (BT, H, N, N)

        # M = w_0 * S^0
        M = ws[:, 0].view(1, H, 1, 1) * P                         # (BT, H, N, N)

        # 方便广播乘法的 S: (BT, 1, N, N)
        S_expand = S.unsqueeze(1)                                  # (BT, 1, N, N)

        # 递推 P = S^k，k = 1..K
        for k in range(1, K + 1):
            # (BT,1,N,N) @ (BT,H,N,N) -> (BT,H,N,N)（batch + head 维度广播）
            P = torch.matmul(S_expand, P)
            M = M + ws[:, k].view(1, H, 1, 1) * P

        # 转成 log-space bias
        bias = torch.full(M.shape, float('-inf'),
                          device=M.device, dtype=torch.float32)   # (BT,H,N,N)
        reachable = M > 0
        if reachable.any():
            vals = torch.log(M[reachable].to(torch.float32) + eps)
            vals = vals.clamp(-clamp_val, clamp_val)
            bias[reachable] = vals

        # 展开 head 维度： (BT, H, N, N) -> (BT*H, N, N)
        bias = bias.view(BT * H, N, N)
        return bias


    def forward(self, x_bt: torch.Tensor, Sf_bt: torch.Tensor, Sb_bt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_bt:  (B*T, N, D)
            Sf_bt: (B*T, N, N) 前向随机游走
            Sb_bt: (B*T, N, N) 后向随机游走
        Returns:
            (B*T, N, D)
        """
        # Pre-Norm
        xin = self.norm1(x_bt)

        # 构建“每个 head 自己一套”的多跳先验 bias
        bias_fw = self._build_multihop_mixture(Sf_bt, self.K, self.alpha_fw, self.eps, self.clamp_val)
        bias_bw = self._build_multihop_mixture(Sb_bt, self.K, self.alpha_bw, self.eps, self.clamp_val)
        # 此时 bias_fw / bias_bw 形状已经是 (B*T * h, N, N)

        attn_bias_fw = bias_fw.to(dtype=xin.dtype, device=xin.device)
        attn_bias_bw = bias_bw.to(dtype=xin.dtype, device=xin.device)

        # MHA
        out_fw, _ = self.attn_fw(xin, xin, xin, attn_mask=attn_bias_fw, need_weights=False)
        out_bw, _ = self.attn_bw(xin, xin, xin, attn_mask=attn_bias_bw, need_weights=False)
        
        # 方向融合
        out_sym  = 0.5 * (out_fw + out_bw)
        out_anti = 0.5 * (out_fw - out_bw)
        gate = torch.sigmoid(self.dir_gate)
        attn_out = self.merge_dropout(self.lin_sym(out_sym) + gate * self.lin_anti(out_anti))

        # 残差 1
        y = x_bt + attn_out

        # FFN + 残差 2
        y = y + self.ffn(self.norm2(y))
        return y


class SpatialKHopModule(nn.Module):
    """
    统一空间建模模块（可选 diffusion 或 attnA 模式，可设置层数）
    """
    def __init__(self, config: Config, c_in: int, c_out: int):
        super().__init__()
        self.config = config
        self.mode = config.spatial_mode
        self.num_layers = config.spatial_num_layers

        if self.mode == "diffusion":
            layers = []
            layers.append(_DiffusionGConvLayer(c_in, c_out, K=config.spatial_K,
                                               dropout=config.st_dropout,
                                               activation=config.activation))
            for _ in range(1, self.num_layers):
                layers.append(_DiffusionGConvLayer(c_out, c_out, K=config.spatial_K,
                                                   dropout=config.st_dropout,
                                                   activation=config.activation))
            self.layers = nn.ModuleList(layers)

        else:  # attnA
            self.in_proj = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
            layers = []
            for _ in range(self.num_layers):
                layers.append(_AttnKHopBiasLayer(dim=c_out, K=config.spatial_K, 
                                                 num_heads=config.spatial_num_heads,
                                                 dropout=config.st_dropout, 
                                                 clamp_val=config.spatial_clamp_val, 
                                                 eps=config.spatial_eps,
                                                 activation=config.activation))
            self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, W_f: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, C_in)
            W_f: (B, N, N)
            W_b: (B, N, N)
        Returns:
            (B, T, N, C_out)
        """
        B, T, N, _ = x.shape

        if self.mode == "diffusion":
            out = x
            for layer in self.layers:
                out = layer(out, W_f, W_b)
            return out

        else:  # attnA
            # 合并时间维度
            x_bt = x.reshape(B * T, N, x.shape[-1])
            x_bt = self.in_proj(x_bt)

            # 预备随机游走核
            Sf = row_normalize(W_f, eps=self.config.spatial_eps)
            Sb = row_normalize(W_b, eps=self.config.spatial_eps)
            Sf_bt = Sf.repeat_interleave(T, dim=0)
            Sb_bt = Sb.repeat_interleave(T, dim=0)

            # 堆叠多层
            h = x_bt
            for layer in self.layers:
                h = layer(h, Sf_bt, Sb_bt)

            # 还原形状
            return h.view(B, T, N, -1)


# ====================================================================================
#                         ST-Block（时空卷积块）
# ====================================================================================

class STBlock(nn.Module):
    """
    时空卷积块: T-Conv -> Spatial -> T-Conv
    统一使用 keep_time=True（时间维度不变）
    """
    def __init__(self, config: Config, c_in: int, c_mid: int, c_out: int):
        super().__init__()
        self.t = TemporalConvGLU(c_in, c_mid, k=config.temporal_kernel, 
                                  dropout=config.st_dropout, config=config)
        self.g = SpatialKHopModule(config, c_mid, c_mid)
        self.t2 = TemporalConvGLU(c_mid, c_mid, k=config.temporal_kernel, 
                                  dropout=config.st_dropout, config=config)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, Wf: torch.Tensor, Wb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, T, N, C_in)
            Wf: (B, N, N)
            Wb: (B, N, N)
        Returns:
            (B, T, N, C_out)
        """
        x = self.t(x)
        x = self.g(x, Wf, Wb)
        x = self.t2(x)
        x = self.norm(x)
        return x


# ====================================================================================
#                    编码器/解码器块（Down/Up）
# ====================================================================================

class DownBlock(nn.Module):
    """
    下采样块: STBlock(same) + 池化(stride=2)
    返回: x_down, skip_feat, skip_T
    """
    def __init__(self, config: Config, c_in: int, c_mid: int, c_out: int):
        super().__init__()
        self.st = STBlock(config, c_in, c_mid, c_out)
        self.pool = nn.Conv1d(c_out, c_out, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor, Wf: torch.Tensor, Wb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            x: (B, T, N, C_in)
        Returns:
            x_down: (B, T_down, N, C_out)
            skip:   (B, T_skip, N, C_out)
            T_skip: int
        """
        B, T, N, C = x.shape
        y = self.st(x, Wf, Wb)  # (B, T', N, C_out)
        skip = y
        T_skip = y.shape[1]

        # 下采样
        BN, C_out, T_prime = B * N, y.shape[-1], y.shape[1]
        y1 = y.permute(0, 2, 3, 1).reshape(BN, C_out, T_prime)
        if T_prime % 2 == 1:
            y1 = F.pad(y1, (0, 1), mode='replicate')
        y1 = self.pool(y1)
        T_down = y1.shape[-1]
        y1 = y1.reshape(B, N, C_out, T_down).permute(0, 3, 1, 2)
        return y1, skip, T_skip


class UpBlock(nn.Module):
    """
    上采样块: 上采样(stride=2) -> 调到 skip_T -> 拼接 -> STBlock(same)
    """
    def __init__(self, config: Config, c_in: int, c_skip: int, c_mid: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(c_in, c_in, kernel_size=2, stride=2)
        self.st = STBlock(config, c_in + c_skip, c_mid, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, skip_T: int, 
                Wf: torch.Tensor, Wb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T_low,  N, C_in)
            skip: (B, T_skip, N, C_skip)
            skip_T: int
        Returns:
            (B, T_skip, N, C_out)
        """
        B, T_low, N, C_in = x.shape
        
        # 上采样
        y = x.permute(0, 2, 3, 1).reshape(B * N, C_in, T_low)
        y = self.up(y)
        y = _adjust_time_1d(y, skip_T)
        y = y.reshape(B, N, C_in, skip_T).permute(0, 3, 1, 2)

        # 拼接 skip
        y = torch.cat([y, skip], dim=-1)

        # STBlock
        y = self.st(y, Wf, Wb)
        return y


# ====================================================================================
#                         主模型：Hourglass-STGCN
# ====================================================================================

class OurAnomalyDetection(nn.Module):
    """
    完整模型: LaneFusion -> 动态层数的 Hourglass-STGCN
    
    结构:
        Input: (B, n_days, T, N, L, F)
        ↓
        LaneFusion: (abn, norm) -> ResidualDiffFusion -> (B, T, N, D_in)
        ↓
        Encoder: Down × num_encoder_blocks
        ↓
        Bottleneck: STBlock
        ↓
        Decoder: Up × num_encoder_blocks
        ↓
        Output: (B, T, N, C_out)
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # LaneFusion 模块
        self.lane_fusion = LaneFusionModule(config)
        
        # 获取通道维度配置
        dims = config.get_channel_dims()  # [D0, D1, D2, ..., D_n]
        num_blocks = config.num_encoder_blocks

        # 初始融合
        self.init_fuse = ResidualDiffFusion(
            dims[0], 
            dropout=config.st_dropout, 
            activation=config.activation
        )

        # 编码器（Down blocks）
        self.down_blocks = nn.ModuleList()
        for i in range(num_blocks):
            c_in = dims[i]
            c_out = dims[i + 1]
            c_mid = c_out  # 中间维度与输出相同
            self.down_blocks.append(DownBlock(config, c_in, c_mid, c_out))

        # 每一层 Up 之前对 skip 的融合
        self.skip_fuse = nn.ModuleList([
            ResidualDiffFusion(dims[i+1], dropout=config.st_dropout, activation=config.activation)
            for i in range(config.num_encoder_blocks - 1, -1, -1)  # 从深到浅
        ])

        # 瓶颈层
        c_bottle = dims[num_blocks]
        self.bottleneck = STBlock(config, c_bottle, c_bottle, c_bottle)

        # 解码器（Up blocks）
        self.up_blocks = nn.ModuleList()
        for i in range(num_blocks - 1, -1, -1):
            c_in = dims[i + 1]      # 上采样输入通道
            c_skip = dims[i + 1]    # skip 连接通道
            c_out = dims[i]         # 输出通道
            c_mid = c_out           # 中间维度
            self.up_blocks.append(UpBlock(config, c_in, c_skip, c_mid, c_out))

        # 输出头
        self.head = nn.Conv2d(in_channels=dims[0], out_channels=config.n_anomaly_types, kernel_size=1)

    def forward(self, traffic: torch.Tensor, mask: torch.Tensor, 
                adj_forward: torch.Tensor, adj_backward: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traffic:      (B, n_days, T, N, L, F)
            mask:         (B, n_days, T, N, L)
            adj_forward:  (B, N, N)
            adj_backward: (B, N, N)
        Returns:
            logits: (B, T, N, C_out)
        """
        # LaneFusion 返回 (abn, norm)
        x_abn, x_norm = self.lane_fusion(traffic, mask)
        B, T0, N, _ = x_abn.shape

        # 初始融合
        x = self.init_fuse(x_abn, x_norm)  # (B, T, N, D)

        # === Encoder（下采样）===
        skip_abn, skip_norm, skip_times = [], [], []
        xa, xn = x_abn, x_norm
        for down_block in self.down_blocks:
            if xn is not None:
                xa_xn = torch.cat([xa, xn], dim=0)                  # (2B,T,N,C)
                Wf2 = torch.cat([adj_forward, adj_forward], dim=0)  # (2B,N,N)
                Wb2 = torch.cat([adj_backward, adj_backward], dim=0)
                y, skip, T_skip = down_block(xa_xn, Wf2, Wb2)       # 只跑一次
                xa, xn = y[:B], y[B:]
                skip_a, skip_n = skip[:B], skip[B:]
            else:
                xa, skip_a, T_skip = down_block(xa, adj_forward, adj_backward)
                skip_n = None
                
            skip_abn.append(skip_a)
            skip_times.append(T_skip)
            if skip_n is not None:
                skip_norm.append(skip_n)

        # === Bottleneck ===
        x = self.bottleneck(xa, adj_forward, adj_backward)

        # === Decoder（上采样）===
        for i, up_block in enumerate(self.up_blocks):
            # 从后往前取 skip
            idx = len(skip_abn) - 1 - i
            skip_a = skip_abn[idx]
            skip_n = skip_norm[idx] if x_norm is not None else None

            # 先做 skip 融合，再喂给 UpBlock
            skip_fused = self.skip_fuse[i](skip_a, skip_n)
            skip_T = skip_times[idx]
            x = up_block(x, skip_fused, skip_T, adj_forward, adj_backward)

        # === 输出头 ===
        # (B, T, N, D) -> (B*N, D, T, 1) -> Conv2d -> (B*N, C, T, 1)
        B, Tfin, N, D = x.shape
        feat = x.permute(0, 2, 1, 3).reshape(B * N, Tfin, D).permute(0, 2, 1).unsqueeze(-1)
        logits = self.head(feat).squeeze(-1)  # (B*N, C, Tfin)
        logits = logits.permute(0, 2, 1).reshape(B, N, Tfin, self.config.n_anomaly_types).permute(0, 2, 1, 3)

        # 对齐到原始时间步
        logits = _adjust_time_4d(logits, target_T=T0)  # (B, T0, N, C)
        return logits
