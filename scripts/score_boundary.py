import numpy as np

def get_assignments(y, yhat, tolerance=1):
    matches = dict((i, []) for i in range(len(yhat)))
    for i, yhat_i in enumerate(yhat):
        dists = np.abs(y - yhat_i)
        idxs = np.argsort(dists)
        for idx in idxs:
            if dists[idx] <= tolerance:
                matches[i].append(idx)
    return matches


def get_counts(y, yhat):
    match_counter = 0
    dup_counter = 0
    miss_counter = 0
    used_idxs = []
    matches = get_assignments(y, yhat)
    dup_frames = []
    miss_frames = []

    for m, vs in matches.items():
        if len(vs) == 0:
            miss_frames.append(m)
            miss_counter += 1
            continue
        vs = sorted(vs)
        dup = False
        for v in vs:
            if v in used_idxs:
                dup = True
            else:
                dup = False
                used_idxs.append(v)
                match_counter += 1
                break
        if dup:
            dup_counter += 1
            dup_frames.append(m)

    return match_counter, dup_counter


def process_segment_predictions(
    hypos,
    refs,
    scores,
    res_files,
): 
    pred_b_len = 0
    b_len = 0
    p_count = 0
    r_count = 0
    p_dup_count = 0
    r_dup_count = 0
    labels = []
    for i, (hypo, ref, score) in enumerate(zip(hypos, refs, scores)):
        hyp_segs = hypo.cumsum(-1)
        if cfg.margin > 0:
            score = score.squeeze(-1)
            skip = (score >= 0.5-cfg.margin) * (score <= 0.5+cfg.margin)
            hyp_segs[skip] = -1
        hyp_segs = list(map(str, hyp_segs.cpu().tolist()))

        ref_segs = ref.cumsum(-1).cpu().tolist()
        ref_segs = list(map(str, ref_segs))
        
        to_write = {}
        to_write[res_files["hypo.segments"]] = " ".join(hyp_segs)
        to_write[res_files["ref.segments"]] = " ".join(ref_segs)

        yhat = (hypo == 1).nonzero().squeeze(-1).cpu().numpy()
        y = (ref == 1).nonzero().squeeze(-1).cpu().numpy()
        b_len += len(y)
        pred_b_len += len(yhat)
        p, pd = get_counts(y, yhat)
        p_count += p
        p_dup_count += pd
        r, rd = get_counts(yhat, y)
        r_count += r
        r_dup_count += rd

        labels.append(to_write)

    for l in labels:
        for dest, label in l.items():
            print(label, file=dest)
            dest.flush()

    return p_count, p_dup_count, r_count, r_dup_count, pred_b_len, b_len




# TODO

    boundary_precision_harsh = None
    boundary_recall_harsh = None
    boundary_precision_lenient = None
    boundary_recall_lenient = None
    if gen_result.pred_b_len > 0:
        boundary_precision_harsh = gen_result.p_count * 100.0 / gen_result.pred_b_len
        boundary_precision_lenient = (gen_result.p_count + gen_result.p_dup_count) * 100.0 / gen_result.pred_b_len
        logger.info(f"Boundary precision (harsh): {boundary_precision_harsh}")
        logger.info(f"Boundary precision (lenient): {boundary_precision_lenient}")

    if gen_result.b_len > 0:
        boundary_recall_harsh = gen_result.r_count * 100.0 / gen_result.b_len
        boundary_recall_lenient = (gen_result.r_count + gen_result.r_dup_count) * 100.0 / gen_result.b_len

