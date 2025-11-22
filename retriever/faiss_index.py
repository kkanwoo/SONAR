#!/usr/bin/env python3
import os, argparse, yaml, numpy as np, random
from pathlib import Path
import faiss

SEED = 1234  # ★ Fix global seed

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    # Complete seed fixing for faiss internal kmeans is limited in the public API, but
    # external factors like sampling/ordering can be reproduced with this seed.

def load_cfg(p):
    with open(Path(p).expanduser(), "r") as f:
        return yaml.safe_load(f)

def build_flat(d, metric):
    return faiss.IndexFlatIP(d) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

def main():
    set_seeds(SEED)  # ★ Set seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="retriever.yaml")
    ap.add_argument("--embeds", default="~/SONAR/results/embeds.npy")
    ap.add_argument("--out", default="~/SONAR/results/faiss.index")
    ap.add_argument("--reuse-index", default=None,
                    help="Trained IVF/PQ index to reuse (codebook). If set, skip train().")
    ap.add_argument("--save-trained", default=None,
                    help="Save trained IVF/PQ index (after train, before add). For clean pipeline.")
    ap.add_argument("--train-sample", type=int, default=0,
                    help="If >0, random subset size for training (speeds up training).")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    x = np.load(Path(args.embeds).expanduser()).astype("float32")
    d = x.shape[1]

    metric_name = str(cfg.get("faiss", {}).get("metric", "ip")).lower()
    metric = faiss.METRIC_INNER_PRODUCT if metric_name == "ip" else faiss.METRIC_L2
    index_type = str(cfg.get("faiss", {}).get("type", "FLAT")).upper()

    outp = Path(args.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)

    if index_type == "FLAT":
        cpu = build_flat(d, metric)
        # GPU add (optional)
        try:
            res = faiss.StandardGpuResources()
            gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
            gpu.add(x)
            cpu = faiss.index_gpu_to_cpu(gpu)
        except Exception:
            cpu.add(x)

    elif index_type in ["IVFPQ", "IVF-PQ", "IVFPQFASTSCAN"]:
        nlist = int(cfg.get("faiss", {}).get("nlist", 4096))
        pq_m  = int(cfg.get("faiss", {}).get("pq_m", 64))

        if args.reuse_index:
            cpu = faiss.read_index(str(Path(args.reuse_index).expanduser()))
            assert cpu.is_trained, "reuse-index is not trained."
            assert cpu.d == d, f"Dim mismatch: reuse-index d={cpu.d}, embeds d={d}"
        else:
            cpu = faiss.index_factory(d, f"IVF{nlist},PQ{pq_m}", metric)
            assert not cpu.is_trained, "Index unexpectedly already trained."

            train_x = x
            if args.train_sample and args.train_sample < x.shape[0]:
                idx = np.random.choice(x.shape[0], args.train_sample, replace=False)
                train_x = x[idx]

            cpu.train(train_x)

            if args.save_trained:
                trained_path = Path(args.save_trained).expanduser()
                trained_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(cpu, str(trained_path))
                print(f"[INFO] saved trained IVF/PQ codebook → {trained_path}")

        # add
        try:
            res = faiss.StandardGpuResources()
            gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
            gpu.add(x)
            cpu = faiss.index_gpu_to_cpu(gpu)
        except Exception:
            cpu.add(x)

    else:
        raise NotImplementedError("Supported types: FLAT, IVFPQ/IVF-PQ/IVFPQFASTSCAN")

    faiss.write_index(cpu, str(outp))
    print("[NOTE] Search parameters like nprobe/efSearch should be set at 'search time'.")
    print(f"[DONE] built index: {cpu.ntotal} vectors -> {outp}")

if __name__ == "__main__":
    main()
