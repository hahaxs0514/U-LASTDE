"""Evaluation metrics used by U-LASTDE."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(
    predictions,
    labels,
    adjs,
    threshold=0.18,
    time_windows=(1, 3, 5, 8),
    core_atol=1e-6,
    threshold_mode="threshold",
    detail_level="compact",
):
    """
    闃堝€硷細
      - threshold: 浣跨敤缁欏畾闃堝€硷紙鏍囬噺=鍏辩敤锛涙垨 len==C 鐨勬暟缁?閫愰€氶亾锛?      - auto:      閫愰€氶亾鏍规嵁鏍囩姝ｆ牱鏈瘮渚?p_c锛屽湪 scores_nc[:,c] 鐨勫垎甯冧笂鍙?(1-p_c) 鍒嗕綅浣滈槇鍊?                   锛堢洰鏍囷細棰勬祴姝ｄ緥姣斾緥鈮堟爣绛炬瘮渚嬶紱鍙楀苟鍒楀€煎奖鍝嶄负杩戜技锛?    鑱氬悎锛?      - classification.overall: 鎸夆€滄瘡閫氶亾鏍囩姝ｆ牱鏈暟鈥濆姞鏉冿紙鏉冮噸 = pos_c / sum(pos)锛?      - spatiotemporal_hit_rate.overall: 鍚屾牱鎸変笂杩版潈閲嶅鍚勯€氶亾鐨?rates 鍔犳潈
    涓嶅啀杈撳嚭 micro/macro銆?    """
    # ---------------- 鍩虹寮犻噺涓庢鏌?----------------
    pred = np.asarray(predictions, dtype=np.float32)
    lab  = np.asarray(labels,      dtype=np.float32)
    A    = np.asarray(adjs,        dtype=np.float32)
    assert pred.shape == lab.shape and pred.ndim == 4, "predictions/labels 蹇呴』鏄?(N,T,S,C)"
    N, T, S, C = pred.shape
    K = len(time_windows)
    tw = np.asarray(time_windows, dtype=int)
    assert K >= 1, "time_windows must contain at least one item"

    # ---------------- 閭绘帴鏃犲悜浜屽€?+ BFS 璺濈 ----------------
    A_sym_bin = (np.maximum(A, np.transpose(A, (0,2,1))) > 0).astype(np.uint8)

    def bfs_all_pairs(adj_bin):
        from collections import deque
        S_ = adj_bin.shape[0]
        dist = np.full((S_, S_), np.inf, dtype=np.float32)
        for s0 in range(S_):
            d = np.full(S_, np.inf, dtype=np.float32)
            d[s0] = 0.0
            q = deque([s0])
            while q:
                u = q.popleft()
                for v in np.nonzero(adj_bin[u])[0]:
                    if d[v] == np.inf:
                        d[v] = d[u] + 1.0
                        q.append(v)
            dist[s0] = d
        return dist

    dists = np.stack([bfs_all_pairs(A_sym_bin[n]) for n in range(N)], axis=0)  # (N,S,S)

    # ---------------- 鍥炲綊鎸囨爣锛堝師鏍蜂繚鐣欙級 ----------------
    mse_overall = float(np.mean((pred - lab) ** 2))
    mae_overall = float(np.mean(np.abs(pred - lab)))
    mse_per_class = [float(np.mean((pred[..., c] - lab[..., c]) ** 2)) for c in range(C)]
    mae_per_class = [float(np.mean(np.abs(pred[..., c] - lab[..., c]))) for c in range(C)]
    regression = {"overall": {"mse": mse_overall, "mae": mae_overall}}
    if detail_level == "detailed":
        regression["per_class"] = [{"mse": mse_per_class[c], "mae": mae_per_class[c]} for c in range(C)]

    # ---------------- 鏍锋湰脳閫氶亾鏍囩/鍒嗘暟 ----------------
    core_mask = np.isclose(lab, 1.0, atol=core_atol)        # (N,T,S,C)
    scores_nc = np.max(pred, axis=(1,2))                    # (N, C)
    labels_nc = core_mask.any(axis=(1,2)).astype(np.uint8)  # (N, C)
    pos_counts_per_class = labels_nc.sum(axis=0).astype(np.int64)  # (C,)
    total_pos_counts = int(pos_counts_per_class.sum())

    # ---------------- 闃堝€硷細threshold | auto ----------------
    if threshold_mode == "threshold":
        if np.isscalar(threshold):
            thr_per_class = np.full((C,), float(threshold), dtype=np.float32)
        else:
            thr_arr = np.asarray(threshold, dtype=np.float32)
            assert thr_arr.shape == (C,), "threshold 涓哄簭鍒楁椂闀垮害蹇呴』绛変簬 C"
            thr_per_class = thr_arr
        thr_info = {"mode": "threshold", "thr_per_class": thr_per_class.tolist()}
    elif threshold_mode == "auto":
        thr_per_class = np.zeros((C,), dtype=np.float32)
        eps = np.finfo(np.float32).eps
        for c in range(C):
            p_true = float(pos_counts_per_class[c]) / float(N)  # 鏍锋湰绾ф渚嬫瘮渚?            sc = scores_nc[:, c]
            if p_true <= 0.0:
                thr_per_class[c] = float(sc.max() + eps)  # 棰勬祴鍏ㄨ礋
            elif p_true >= 1.0:
                thr_per_class[c] = float(sc.min() - eps)  # 棰勬祴鍏ㄦ
            else:
                thr_per_class[c] = float(np.quantile(sc, 1.0 - p_true))
        thr_info = {
            "mode": "auto",
            "target_positive_ratio_per_class": (
                (pos_counts_per_class / max(N,1)).astype(np.float32).tolist()
            ),
            "thr_per_class": thr_per_class.tolist()
        }
    else:
        raise ValueError("threshold_mode 鍙兘涓?'threshold' 鎴?'auto'")

    # ---------------- 鍒嗙被鎸囨爣锛歱er-class + overall(鍔犳潈) ----------------
    def _safe_auc_roc(y_true, y_score):
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return float("nan")

    def _safe_ap(y_true, y_score):
        try:
            return float(average_precision_score(y_true, y_score))
        except Exception:
            return float("nan")

    per_class_metrics = []
    for c in range(C):
        y_true = labels_nc[:, c]
        y_score = scores_nc[:, c]
        thr_c = thr_per_class[c]
        y_pred = (y_score >= thr_c).astype(np.uint8)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        per_class_metrics.append({
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": _safe_auc_roc(y_true, y_score),
            "pr_auc": _safe_ap(y_true, y_score),
            "threshold": float(thr_c),
            "pos_count": int(pos_counts_per_class[c])
        })

    # Overall metrics are weighted by positive sample counts per class.
    def _weighted_overall(per_class_list, weights_raw):
        if weights_raw.sum() <= 0:
            # 鏃犳鏍锋湰锛氳繑鍥?NaN
            return {k: float("nan") for k in ["precision","recall","f1","roc_auc","pr_auc"]}
        w = weights_raw.astype(np.float64)
        w = w / w.sum()
        out = {}
        for key in ["precision","recall","f1","roc_auc","pr_auc"]:
            vals = np.array([m[key] for m in per_class_list], dtype=np.float64)
            mask = np.isfinite(vals)
            if not mask.any():
                out[key] = float("nan")
            else:
                w_eff = w.copy()
                w_eff[~mask] = 0.0
                s = w_eff.sum()
                out[key] = float(np.sum((w_eff/s) * vals[mask])) if s>0 else float("nan")
        return out

    cls_overall = _weighted_overall(per_class_metrics, pos_counts_per_class)

    classification = {
        "overall": {
            **cls_overall,
            "info": {
                "thresholding": thr_info,
                "pos_counts_per_class": pos_counts_per_class.tolist(),
                "total_pos_counts": int(total_pos_counts)
            }
        }
    }
    if detail_level == "detailed":
        classification["per_class"] = per_class_metrics

    # ---------------- 瑕嗙洊寮忓懡涓巼锛氶€愰€氶亾 -> overall(鍔犳潈) ----------------
    def _recall_rates_per_class(c_idx):
        thr_c = thr_per_class[c_idx]
        hits = np.zeros(K, dtype=np.int64); total = 0
        for n in range(N):
            dist_n = dists[n]
            G_mask = core_mask[n, :, :, c_idx]                 # (T,S)
            gt_pos = np.argwhere(G_mask)
            if gt_pos.size == 0:
                continue
            P_mask = (pred[n, :, :, c_idx] >= thr_c)           # (T,S)
            if not P_mask.any():
                total += gt_pos.shape[0]
                continue
            P_by_t = [np.nonzero(P_mask[t])[0] for t in range(T)]
            for (t0, s0) in gt_pos:
                total += 1
                for r in range(K):
                    dt = tw[r]
                    t_lo = max(0, t0 - dt); t_hi = min(T - 1, t0 + dt)
                    matched = False
                    for t1 in range(t_lo, t_hi + 1):
                        nodes = P_by_t[t1]
                        if nodes.size == 0: continue
                        if np.any(dist_n[s0, nodes] <= r):
                            matched = True; break
                    if matched:
                        hits[r:] += 1
                        break
        rates = [float(h/total) if total>0 else np.nan for h in hits]
        return {"hits": hits, "total": total, "rates": rates}

    def _precision_rates_per_class(c_idx):
        thr_c = thr_per_class[c_idx]
        hits = np.zeros(K, dtype=np.int64); total = 0
        for n in range(N):
            dist_n = dists[n]
            P_mask = (pred[n, :, :, c_idx] >= thr_c)           # (T,S)
            pred_pos = np.argwhere(P_mask)
            if pred_pos.size == 0:
                continue
            G_mask = core_mask[n, :, :, c_idx]                 # (T,S)
            if not G_mask.any():
                total += pred_pos.shape[0]
                continue
            G_by_t = [np.nonzero(G_mask[t])[0] for t in range(T)]
            for (t1, s1) in pred_pos:
                total += 1
                for r in range(K):
                    dt = tw[r]
                    t_lo = max(0, t1 - dt); t_hi = min(T - 1, t1 + dt)
                    matched = False
                    for t0 in range(t_lo, t_hi + 1):
                        nodes = G_by_t[t0]
                        if nodes.size == 0: continue
                        if np.any(dist_n[s1, nodes] <= r):
                            matched = True; break
                    if matched:
                        hits[r:] += 1
                        break
        rates = [float(h/total) if total>0 else np.nan for h in hits]
        return {"hits": hits, "total": total, "rates": rates}

    # 姣忛€氶亾璁＄畻
    recall_pc    = [_recall_rates_per_class(c)   for c in range(C)]
    precision_pc = [_precision_rates_per_class(c) for c in range(C)]

    # Overall hit rates are weighted by positive sample counts per class.
    def _weighted_rates(per_class_dicts, weights_raw):
        if weights_raw.sum() <= 0:
            return {"rates": [float("nan")]*K}
        w = (weights_raw / weights_raw.sum()).astype(np.float64)
        mat = []
        w_eff = []
        for c, d in enumerate(per_class_dicts):
            rates = np.asarray(d["rates"], dtype=np.float64)
            if np.all(np.isfinite(rates)):
                mat.append(rates)
                w_eff.append(w[c])
            else:
                # 濡傛灉璇ラ€氶亾 rate 缂哄け锛堟瘮濡?total=0锛夛紝瀵瑰簲鏉冮噸浣滃簾
                pass
        if len(mat) == 0:
            return {"rates": [float("nan")]*K}
        mat = np.stack(mat, axis=0)         # (C_eff, K)
        w_eff = np.asarray(w_eff, dtype=np.float64)
        w_eff = w_eff / w_eff.sum()
        rates = (w_eff.reshape(-1, 1) * mat).sum(axis=0)
        return {"rates": rates.tolist()}

    recall_overall    = _weighted_rates(recall_pc, pos_counts_per_class)
    precision_overall = _weighted_rates(precision_pc, pos_counts_per_class)

    spatiotemporal_hit_rate = {
        "recall_style":   {"overall": recall_overall},
        "precision_style":{"overall": precision_overall},
        "info": {
            "thr_per_class": thr_per_class.tolist(),
            "time_windows": tw.tolist(),
            "weights_by_label_pos": pos_counts_per_class.tolist()
        }
    }
    if detail_level == "detailed":
        spatiotemporal_hit_rate["recall_style"]["per_class"] = [
            {"hits": r["hits"].tolist(), "total": int(r["total"]), "rates": r["rates"]}
            for r in recall_pc
        ]
        spatiotemporal_hit_rate["precision_style"]["per_class"] = [
            {"hits": r["hits"].tolist(), "total": int(r["total"]), "rates": r["rates"]}
            for r in precision_pc
        ]

    # ---------------- 姹囨€昏繑鍥?----------------
    out = {
        "config": {
            "detail_level": detail_level,
            "threshold_mode": threshold_mode,
        },
        "regression": regression,
        "classification": classification,
        "spatiotemporal_hit_rate": spatiotemporal_hit_rate
    }
    return out
