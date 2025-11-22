#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, time, csv, re
from collections import defaultdict
from pathlib import Path
import numpy as np

def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)

def load_ids(ids_path):
    with open(ids_path, "r") as f:
        ids = json.load(f)
    return np.array(ids)

def load_faiss_or_build(index_path, embeds_path):
    import faiss
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        info("FAISS index loaded:", index_path)
        return index, None
    xb = np.load(embeds_path)  # (N, d)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    info("Built Flat-IP index on-the-fly from", embeds_path)
    return index, xb

def load_wm_maps(meta_path):
    """
    injection_meta.jsonl -> (id_to_m, m_to_ids, M_est)
    Each line: {"id": "...", "assigned_beacon": m, ...}
    """
    id_to_m = {}
    m_to_ids = defaultdict(set)
    M_est = 0
    with open(os.path.expanduser(meta_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            gid = str(j["id"])
            m = int(j["assigned_beacon"])
            id_to_m[gid] = m
            m_to_ids[m].add(gid)
            if m + 1 > M_est:
                M_est = m + 1
    return id_to_m, m_to_ids, M_est

def parse_m_from_filename(name):
    """
    Expect patterns like: probe_m0.png, any*_m7_*.png
    Returns int(m) or None.
    """
    m = None
    mobj = re.search(r"[._-]m(\d+)", name)
    if not mobj:
        mobj = re.search(r"^probe_m(\d+)", name)
    if mobj:
        m = int(mobj.group(1))
    return m

def build_probe_m_list(probe_names, img_dir, probe_meta_jsonl=None):
    """
    Returns: list[int or None] aligned to probe_names.
    Prefer probe_meta_jsonl (path->m), else parse filename.
    """
    if probe_meta_jsonl:
        path_to_m = {}
        with open(os.path.expanduser(probe_meta_jsonl), "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                m = int(j["m"])
                p = str(j["path"])
                path_to_m[os.path.basename(p)] = m
        return [path_to_m.get(fn, None) for fn in probe_names]
    else:
        return [parse_m_from_filename(fn) for fn in probe_names]

def ensure_q_embeds_text(queries_path, out_path, device="cuda", batch_size=256, query_key="query"):
    """
    TEXT JSONL -> CLIP text embeddings (L2-normalized) -> npy
    """
    if os.path.exists(out_path):
        arr = np.load(out_path)
        info(f"Loaded cached q_embeds (text): {out_path} {arr.shape}")
        return arr, None

    import torch
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"

    # Try openai/clip, fallback to open_clip
    txt_encode = None
    tok_fn = None
    try:
        import clip
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        def _tok(batch): return clip.tokenize(batch, truncate=True).to(device)
        @torch.no_grad()
        def _enc(tokens):
            z = model.encode_text(tokens).float()
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            return z
        tok_fn = _tok; txt_encode = _enc
        info("Using openai/clip ViT-L/14 for text encoding")
    except Exception:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        def _tok(batch): return tokenizer(batch)
        @torch.no_grad()
        def _enc(tokens):
            if hasattr(tokens, "to"): tokens = tokens.to(device)
            z = model.encode_text(tokens).float()
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            return z
        tok_fn = _tok; txt_encode = _enc
        info("Using open-clip ViT-L-14(openai) for text encoding")

    # Load texts
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            key = query_key if (query_key in obj) else "query"
            queries.append(obj[key])
    info("Total text queries:", len(queries))

    # Encode
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            tokens = tok_fn(batch)
            z = txt_encode(tokens)
            all_vecs.append(z.cpu().numpy())
            if (i // batch_size) % 20 == 0:
                info(f"encoded {i}/{len(queries)}")
    qemb = np.concatenate(all_vecs, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, qemb)
    info("Saved q_embeds:", out_path, qemb.shape)
    return qemb, None

def ensure_q_embeds_images(img_dir, out_path, model="ViT-L-14", pretrained="openai", device="cuda", batch_size=256):
    """
    Image directory -> CLIP image embeddings (L2-normalized) -> npy
    Returns: (embeddings, probe_names)
    """
    img_dir = os.path.expanduser(img_dir)
    if os.path.exists(out_path):
        arr = np.load(out_path)
        info(f"Loaded cached q_embeds (images): {out_path} {arr.shape}")
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        return arr, files

    import torch, torch.nn.functional as F
    from PIL import Image
    import open_clip

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"
    model_obj, preprocess_train, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    model_obj.eval()

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    paths = [os.path.join(img_dir, f) for f in files]
    info("Total image probes:", len(paths))

    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            ims = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB")
                ims.append(preprocess(im))
            X = torch.stack(ims, dim=0).to(device)
            z = model_obj.encode_image(X).float()
            z = F.normalize(z, p=2, dim=1)
            vecs.append(z.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                info(f"encoded {i}/{len(paths)}")
    qemb = np.concatenate(vecs, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, qemb)
    info("Saved q_embeds (images):", out_path, qemb.shape)
    return qemb, files

def recall_at_k(ranks, k):
    hits = sum(1 for r in ranks if r is not None and r < k)
    return hits / len(ranks) if ranks else 0.0

def mrr(ranks):
    s = 0.0; n = len(ranks)
    for r in ranks:
        s += 1.0/(r+1) if r is not None else 0.0
    return s/n if n else 0.0

def ndcg_at_k(ranks, k):
    def dcg(rank, k):
        if rank is None or rank >= k: return 0.0
        return 1.0/np.log2(rank+2)
    return float(np.mean([dcg(r,k) for r in ranks])) if ranks else 0.0

# ----------------- WM metrics per-probe -----------------
def _per_probe_metrics(ret_ids, ids_set_w, ids_set_bankaware=None, K_for_ndcg=None):
    """
    ret_ids: list[str] length K (Top-K)
    ids_set_w: set of all WM ids (overall view)
    ids_set_bankaware: set of WM ids for this probe's bank (if bank-aware), else None
    K_for_ndcg: int or None -> nDCG cut-off (defaults to K)
    returns dict with share, count, hit(>=1), fhr, dcg (overall and bank if available)
    """
    K = len(ret_ids)
    Kd = K_for_ndcg or K

    # overall
    rel_overall = [1.0 if r in ids_set_w else 0.0 for r in ret_ids]
    count_overall = int(sum(rel_overall))
    share_overall = float(count_overall) / K
    try:
        fhr_overall = next((i+1 for i,v in enumerate(rel_overall) if v>0), float("inf"))
    except StopIteration:
        fhr_overall = float("inf")
    hit_overall = 1.0 if fhr_overall != float("inf") else 0.0
    dcg_overall = sum(rel_overall[i]/np.log2(i+2) for i in range(min(K, Kd)))

    out = {
        "share_overall": share_overall,
        "count_overall": count_overall,
        "hit_overall": hit_overall,
        "fhr_overall": fhr_overall,
        "dcg_overall": dcg_overall,
    }

    # bank-aware (optional)
    if ids_set_bankaware is not None:
        rel_bank = [1.0 if r in ids_set_bankaware else 0.0 for r in ret_ids]
        count_bank = int(sum(rel_bank))
        share_bank = float(count_bank) / K
        try:
            fhr_bank = next((i+1 for i,v in enumerate(rel_bank) if v>0), float("inf"))
        except StopIteration:
            fhr_bank = float("inf")
        hit_bank = 1.0 if fhr_bank != float("inf") else 0.0
        dcg_bank = sum(rel_bank[i]/np.log2(i+2) for i in range(min(K, Kd)))
        out.update({
            "share_bank": share_bank,
            "count_bank": count_bank,
            "hit_bank": hit_bank,
            "fhr_bank": fhr_bank,
            "dcg_bank": dcg_bank,
        })

    return out
# -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="~/SONAR/eval/queries_general.jsonl",
                    help="Text queries (JSONL). Can be omitted if image queries are used.")
    ap.add_argument("--query-key", default="query",
                    help="Key containing text in text query JSONL (probe uses 'text')")
    ap.add_argument("--query-images", default=None,
                    help="Image query directory (png/jpg). If specified, uses image queries instead of text")

    ap.add_argument("--ids", default="~/SONAR/results/ids.json")
    ap.add_argument("--embeds", default="~/SONAR/results/embeds.npy")
    ap.add_argument("--faiss", default="~/SONAR/results/faiss.index")
    ap.add_argument("--out-metrics", default="~/SONAR/results/metrics.csv")
    ap.add_argument("--out-topk", default="~/SONAR/results/retrieval_topk.jsonl")
    ap.add_argument("--q-embeds", default="~/SONAR/results/q_embeds.npy")

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--mode", choices=["ir", "wm"], default="ir",
                    help="ir: General IR evaluation / wm: Watermark verification (probe)")
    ap.add_argument("--wm-ids", default=None,
                    help="Watermark ID list (txt, one per line). Required for --mode wm")

    # Options for image encoding
    ap.add_argument("--img-model", default="ViT-L-14")
    ap.add_argument("--img-pretrained", default="openai")

    ap.add_argument("--bank-aware", action="store_true",
                help="Enable bank-aware WM evaluation (probe m must match assigned_beacon m of returned WM ids).")

    ap.add_argument("--wm-map", default=None,
                help="Path to injection_meta.jsonl to build id->m and m->ids maps.")
    ap.add_argument("--probe-meta", default=None,
                help="Path to image_probe_meta.jsonl (optional). If provided, probe m will be read from here instead of filename parsing.")
    ap.add_argument("--wsn-thresh", type=int, default=2,
                help="Ownership verification threshold.")
    ap.add_argument("--wsn-operator", choices=["gt","ge"], default="ge",
                help="Ownership decision rule: WSN > thresh (gt) or WSN >= thresh (ge)")

    # Lift / permutation
    ap.add_argument("--lift", action="store_true",
                    help="Compute bank-aware Lift@k and (optionally) permutation p-value.")
    ap.add_argument("--perm", type=int, default=0,
                    help="Permutation count for p-value (e.g., 1000). 0=skip.")

    ap.add_argument("--succ-thresh", default="1",
                help="succ@K thresholds as comma list, e.g., '1,2'. Default '1'.")
    ap.add_argument("--ndcg-k", type=int, default=None,
                help="Override K used in nDCG@K (defaults to --topk).")

    args = ap.parse_args()

    ids_path     = os.path.expanduser(args.ids)
    embeds_path  = os.path.expanduser(args.embeds)
    faiss_path   = os.path.expanduser(args.faiss)
    out_metrics  = os.path.expanduser(args.out_metrics)
    out_topk     = os.path.expanduser(args.out_topk)
    q_emb_path   = os.path.expanduser(args.q_embeds)
    queries_path = os.path.expanduser(args.queries) if args.queries else None
    img_dir      = os.path.expanduser(args.query_images) if args.query_images else None

    ids = load_ids(ids_path)
    id2row = {ids[i]: i for i in range(len(ids))}
    info("IDs loaded:", len(ids))

    index, xb = load_faiss_or_build(faiss_path, embeds_path)
    try:
        index.nprobe = 32
    except Exception:
        pass

    # ----- Query embeddings -----
    probe_names = None
    if img_dir:
        qemb, probe_names = ensure_q_embeds_images(
            img_dir, q_emb_path, model=args.img_model, pretrained=args.img_pretrained,
            device=args.device, batch_size=args.batch
        )
        is_image_queries = True
        info(f"Using IMAGE queries from: {img_dir}")
    else:
        qemb, _ = ensure_q_embeds_text(
            queries_path, q_emb_path, device=args.device, batch_size=args.batch, query_key=args.query_key
        )
        is_image_queries = False
        info(f"Using TEXT queries from: {queries_path}")

    # --------- WM mode ---------
    if args.mode == "wm":
        assert args.wm_ids is not None, "--wm-ids is required for --mode wm."
        wm_ids = set(l.strip() for l in open(os.path.expanduser(args.wm_ids), "r") if l.strip())
        info("WM IDs loaded:", len(wm_ids))

        import faiss
        K = int(args.topk)
        K_ndcg = int(args.ndcg_k) if args.ndcg_k is not None else K
        succ_thresholds = [int(s) for s in str(args.succ_thresh).split(",") if s.strip()]
        Q = qemb.shape[0]

        # Probes (for logging)
        if not is_image_queries:
            probes = [json.loads(l.strip()) for l in open(queries_path, "r", encoding="utf-8") if l.strip()]
        else:
            probes = probe_names  # filenames

        # Bank-aware maps
        id_to_m = None; m_to_ids = None; M_est = None
        probe_m_list = None
        if args.bank_aware:
            assert args.wm_map is not None, "--wm-map (injection_meta.jsonl) is required for --bank-aware mode."
            id_to_m, m_to_ids, M_est = load_wm_maps(args.wm_map)
            info(f"Bank-aware mapping loaded: {len(id_to_m)} ids across ~{M_est} beacons")

            if is_image_queries:
                probe_m_list = build_probe_m_list(probe_names, img_dir, args.probe_meta)
                n_none = sum(1 for v in probe_m_list if v is None)
                if n_none > 0:
                    warn(f"[bank-aware] {n_none} probes have unknown m. They are excluded from bank-aware-only summaries.")
            else:
                probe_m_list = []
                missing = 0
                for ex in probes:
                    mm = ex.get("m", ex.get("beacon", None))
                    if mm is None:
                        missing += 1
                        probe_m_list.append(None)
                    else:
                        probe_m_list.append(int(mm))
                if missing > 0:
                    warn(f"[bank-aware] {missing} text probes have no m/beacon; excluded from bank-aware-only summaries.")

        # Aggregation containers
        shares_overall, dcgs_overall, fhrs_overall, hits_overall = [], [], [], []
        succ_counts_overall = {t: 0 for t in succ_thresholds}

        shares_bank, dcgs_bank, fhrs_bank, hits_bank = [], [], [], []
        succ_counts_bank = {t: 0 for t in succ_thresholds}

        rets = []
        BATCH = 1024
        n_done = 0
        t0 = time.time()

        # Optional confusion (beacon×beacon)
        confusion = None
        if args.bank_aware and M_est and M_est > 0:
            confusion = np.zeros((M_est, M_est), dtype=np.int64)

        with open(out_topk, "w", encoding="utf-8") as fw:
            for i in range(0, Q, BATCH):
                QV = qemb[i:i+BATCH].astype(np.float32)
                D, I = index.search(QV, K)
                for j in range(len(QV)):
                    rows = I[j].tolist()
                    ret_ids = [ids[r] for r in rows]
                    rets.append(ret_ids)

                    # Per-probe sets
                    s_overall = wm_ids
                    ids_bank = None
                    m_probe = None
                    if args.bank_aware and probe_m_list is not None:
                        m_probe = probe_m_list[i+j]
                        if m_probe is not None:
                            ids_bank = m_to_ids.get(m_probe, set())

                    # Per-probe metrics
                    mvals = _per_probe_metrics(ret_ids, s_overall, ids_bank, K_for_ndcg=K_ndcg)
                    shares_overall.append(mvals["share_overall"])
                    dcgs_overall.append(mvals["dcg_overall"])
                    fhrs_overall.append(mvals["fhr_overall"])
                    hits_overall.append(mvals["hit_overall"])

                    # succ@K(t): overall
                    wm_count_overall = mvals["count_overall"]
                    for t in succ_thresholds:
                        if wm_count_overall >= t:
                            succ_counts_overall[t] += 1

                    # bank-aware (optional)
                    if ids_bank is not None:
                        shares_bank.append(mvals.get("share_bank", 0.0))
                        dcgs_bank.append(mvals.get("dcg_bank", 0.0))
                        fhrs_bank.append(mvals.get("fhr_bank", float("inf")))
                        hits_bank.append(mvals.get("hit_bank", 0.0))

                        wm_count_bank = mvals.get("count_bank", 0)
                        for t in succ_thresholds:
                            if wm_count_bank >= t:
                                succ_counts_bank[t] += 1

                    # confusion
                    if confusion is not None and m_probe is not None:
                        found_ms = set()
                        for g in ret_ids:
                            mm = id_to_m.get(g, None)
                            if mm is not None:
                                found_ms.add(mm)
                        for mm in found_ms:
                            if 0 <= m_probe < confusion.shape[0] and 0 <= mm < confusion.shape[1]:
                                confusion[m_probe, mm] += 1

                    # Output record
                    if is_image_queries:
                        probe_repr = probes[i+j]  # filename
                    else:
                        ex = probes[i+j]
                        probe_repr = ex.get(args.query_key)

                    rec = {
                        "probe": probe_repr,
                        "topk_ids": ret_ids,
                        "topk_scores": D[j].tolist(),
                        "wm_hits_overall": wm_count_overall
                    }
                    if m_probe is not None:
                        rec["m_probe"] = int(m_probe)
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += len(QV)

        info(f"[WM] searched {n_done} probes in {time.time()-t0:.2f}s (K={K})")

        # ---- Aggregate (overall) ----
        Q_total = len(shares_overall)
        over_share = float(np.mean(shares_overall)) if shares_overall else 0.0
        over_hit_rate = float(np.mean(hits_overall)) if hits_overall else 0.0
        # MRR_wm (mean reciprocal of first-hit ranks)
        mrr_wm_over = float(np.mean([0.0 if f == float("inf") else 1.0/f for f in fhrs_overall])) if fhrs_overall else 0.0
        
        # nDCG@K (binary gains, K_ndcg cutoff, corpus-level IDCG_K)
        L = min(K_ndcg, K)
        idcg = sum(1.0/np.log2(i+2) for i in range(L)) if L > 0 else 0.0
        ndcg_over = float(np.mean([d/idcg if idcg > 0 else 0.0 for d in dcgs_overall])) if dcgs_overall else 0.0

        # succ@K(t)
        succ_over = {t: (succ_counts_overall[t]/Q_total if Q_total>0 else 0.0) for t in succ_thresholds}

        # Lift@K (coverage p = |W| / N)
        Ntotal = len(ids)
        p_cov = len(wm_ids) / max(1, Ntotal)
        lift_over = (over_share / p_cov) if p_cov > 0 else float("nan")

        # ---- Aggregate (bank-aware) ----
        bank_present = len(shares_bank) > 0
        if bank_present:
            Q_known = len(shares_bank)
            bank_share = float(np.mean(shares_bank))
            bank_hit_rate = float(np.mean(hits_bank))
            mrr_wm_bank = float(np.mean([0.0 if f == float("inf") else 1.0/f for f in fhrs_bank])) if fhrs_bank else 0.0
            ndcg_bank = float(np.mean([d/idcg if idcg > 0 else 0.0 for d in dcgs_bank])) if dcgs_bank else 0.0
            succ_bank = {t: (succ_counts_bank[t]/Q_known if Q_known>0 else 0.0) for t in succ_thresholds}
            lift_bank = (bank_share / p_cov) if p_cov > 0 else float("nan")
        else:
            Q_known = 0
            bank_share = bank_hit_rate = mrr_wm_bank = ndcg_bank = float("nan")
            succ_bank = {t: float("nan") for t in succ_thresholds}
            lift_bank = float("nan")

        # Save confusion (optional)
        if confusion is not None:
            conf_out = os.path.splitext(out_metrics)[0].replace(".csv","") + "_confusion_bankaware.csv"
            np.savetxt(conf_out, confusion, fmt="%d", delimiter=",")
            info(f"Saved bank-aware confusion matrix → {conf_out}")

        # Save CSV (summary)
        Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(out_metrics, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            succ_cols_over = [f"succ@K_overall(t={t})" for t in succ_thresholds]
            succ_cols_bank = [f"succ@K_bankaware(t={t})" for t in succ_thresholds]
            header = [
                "mode","K","N_probes","N_known_m","coverage_p",
                "share@K_overall","hit_rate_overall","MRR_wm_overall","nDCG@K_overall","Lift@K_overall",
                *succ_cols_over,
                "share@K_bankaware","hit_rate_bankaware","MRR_wm_bankaware","nDCG@K_bankaware","Lift@K_bankaware",
                *succ_cols_bank
            ]
            w.writerow(header)
            row = [
                "WM", K, Q_total, Q_known, p_cov,
                over_share, over_hit_rate, mrr_wm_over, ndcg_over, lift_over,
                *[succ_over[t] for t in succ_thresholds],
                bank_share, bank_hit_rate, mrr_wm_bank, ndcg_bank, lift_bank,
                *[succ_bank[t] for t in succ_thresholds]
            ]
            w.writerow(row)

        # Console log
        print(f"[OVERALL-W] share@{K}: {over_share:.3f} | hit@{K}: {over_hit_rate:.3f} | MRR_wm: {mrr_wm_over:.3f} | nDCG@{K_ndcg}: {ndcg_over:.3f} | Lift: {lift_over:.2f}")
        if bank_present:
            print(f"[BANK-AWARE] share@{K}: {bank_share:.3f} | hit@{K}: {bank_hit_rate:.3f} | MRR_wm: {mrr_wm_bank:.3f} | nDCG@{K_ndcg}: {ndcg_bank:.3f} | Lift: {lift_bank:.2f}")
        for t in succ_thresholds:
            so = succ_over[t]
            sb = succ_bank[t] if bank_present else float("nan")
            print(f"succ@{K}({t}): overall={so:.3f}, bank-aware={sb:.3f}")
        return

    # --------- IR mode ---------
    import faiss
    K = args.topk
    ranks_all = []
    ranks_webqa, ranks_mmqa = [], []

    with open(out_topk, "w", encoding="utf-8") as fw:
        n = 0
        t0 = time.time()
        B = 1024

        exs = [json.loads(line) for line in open(queries_path, "r", encoding="utf-8")]
        for i in range(0, len(exs), B):
            batch = exs[i:i+B]
            QV = qemb[i:i+B]
            D, I = index.search(QV.astype(np.float32), K)
            for j, ex in enumerate(batch):
                rel_ids = set(ex["relevant_ids"])
                ds = ex["dataset"]
                rank = None
                for k in range(K):
                    rid = ids[I[j, k]]
                    if rid in rel_ids:
                        rank = k; break
                ranks_all.append(rank)
                (ranks_webqa if ds == "WebQA" else ranks_mmqa).append(rank)
                out = {
                    "dataset": ds,
                    "qid": ex.get("qid"),
                    "query": ex.get("query"),
                    "relevant_ids": list(rel_ids),
                    "topk_ids": [ids[ii] for ii in I[j].tolist()],
                    "topk_scores": D[j].tolist()
                }
                fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        info(f"searched {n} queries in {time.time()-t0:.2f}s")

    def summarize(ranks):
        return {
            "R@1":  recall_at_k(ranks, 1),
            "R@5":  recall_at_k(ranks, 5),
            "R@10": recall_at_k(ranks,10),
            "MRR":  mrr(ranks),
            "nDCG@10": ndcg_at_k(ranks,10)
        }

    m_all   = summarize(ranks_all)
    m_webqa = summarize(ranks_webqa)
    m_mmqa  = summarize(ranks_mmqa)

    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    header = ["split","R@1","R@5","R@10","MRR","nDCG@10","N"]
    rows = [
        ["ALL",   m_all["R@1"], m_all["R@5"], m_all["R@10"], m_all["MRR"], m_all["nDCG@10"], len(ranks_all)],
        ["WebQA", m_webqa["R@1"],m_webqa["R@5"],m_webqa["R@10"],m_webqa["MRR"],m_webqa["nDCG@10"],len(ranks_webqa)],
        ["MMQA",  m_mmqa["R@1"], m_mmqa["R@5"], m_mmqa["R@10"], m_mmqa["MRR"], m_mmqa["nDCG@10"], len(ranks_mmqa)],
    ]
    with open(out_metrics, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); [w.writerow(r) for r in rows]

    info("== Metrics ==")
    for r in rows:
        tag = r[0]
        print(f"{tag:>6s} | R@1 {r[1]:.3f}  R@5 {r[2]:.3f}  R@10 {r[3]:.3f}  MRR {r[4]:.3f}  nDCG@10 {r[5]:.3f}  (N={r[6]})")

if __name__ == "__main__":
    main()
