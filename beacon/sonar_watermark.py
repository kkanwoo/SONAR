#!/usr/bin/env python3
"""
SONAR: Latent Subspace Beacon — Watermark Injection (Bank-Aligned)

Change Summary (Aligned with Verification Query):
- Generate fixed beacon bank {b_m}_{m=0}^{M-1} using secret-seed and salt=m
- Map each image gid to m ∈ {0..M-1} using HMAC(secret, gid)
- v_target = normalize((1 - alpha) * base + alpha * b_m)
  * base is default μ (mean of clean embeddings), can select individual embedding basis with --mix-base e0
- Record assigned beacon index(m) in metadata

Usage (example):
python sonar_watermark.py \
    --index-parquet ~/SONAR/data/index.parquet \
    --ids-json ~/SONAR/results/ids.json \
    --images-root ~/SONAR/data \
    --out-dir ~/SONAR/beacon/out \
    --secret-seed "cvpr2026-secret" \
    --subspace-dim 16 \
    --beacon-bank 32 \
    --embeds-path ~/SONAR/results/embeds.npy \
    --target-frac 0.05 \
    --dataset-filter WebQA,MMQA \
    --eps 4.0 \
    --norm linf \
    --steps 60 \
    --step-size 1.0 \
    --alpha 0.75 \
    --target-cos 0.45 \
    --device cuda \
    --mix-base mu

Notes
- eps/step-size are interpreted in model input space (normalized tensor). (If exact pixel space L∞ guarantee is desired, reconstruction as constraint before preprocessing is needed)
"""
import os, json, hmac, hashlib, argparse, math, random
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# =======================
# CLIP loader (fallback)
# =======================
def load_clip_model(device="cuda"):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # (A) Prioritize openai/clip
    try:
        import clip # pip install git+https://github.com/openai/CLIP.git
        model, preprocess = clip.load("ViT-L/14", device=device)
        model.eval()

        def encode_image_infer(img_tensor):
            with torch.no_grad():
                z = model.encode_image(img_tensor).float()
                z = torch.nn.functional.normalize(z, p=2, dim=1)
                return z

        def encode_image_grad(img_tensor):
            z = model.encode_image(img_tensor).float()
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            return z

        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            dim = model.encode_image(dummy).shape[-1]
        return model, preprocess, encode_image_infer, encode_image_grad, int(dim), device
    except Exception:
        pass

    # (B) open-clip fallback
    try:
        import open_clip # pip install open-clip-torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
        model.eval()

        def encode_image_infer(img_tensor):
            with torch.no_grad():
                z = model.encode_image(img_tensor).float()
                z = torch.nn.functional.normalize(z, p=2, dim=1)
                return z

        def encode_image_grad(img_tensor):
            z = model.encode_image(img_tensor).float()
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            return z

        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            dim = model.encode_image(dummy).shape[-1]
        return model, preprocess, encode_image_infer, encode_image_grad, int(dim), device
    except Exception as e:
        raise RuntimeError(
            "CLIP model not available. Install one of:\n"
            " pip install git+https://github.com/openai/CLIP.git\n"
            " or\n"
            " pip install open-clip-torch"
        ) from e

# =======================
# Secret Subspace (U)
# =======================
def seeded_orthonormal_subspace(embed_dim: int, k: int, secret_seed: str) -> torch.Tensor:
    """ Deterministically generate an orthonormal basis U (embed_dim x k) from secret_seed. """
    digest = hashlib.sha256(secret_seed.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big") % (2**31 - 1)

    rng_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    random_state = random.getstate()

    torch.manual_seed(seed_int)
    np.random.seed(seed_int)
    random.seed(seed_int)

    G = torch.randn(embed_dim, k)
    Q, _ = torch.linalg.qr(G, mode="reduced")

    torch.random.set_rng_state(rng_state)
    np.random.set_state(np_state)
    random.setstate(random_state)

    return Q[:, :k] # (D, k)

# -----------------------
# Beacon bank (fixed M)
# -----------------------
def make_beacon_from_bank(U: torch.Tensor, secret: str, m: int) -> torch.Tensor:
    """ Generate m-th b_m of beacon bank (fixed direction inside U) """
    digest = hashlib.sha256((f"beacon-{secret}-{m}").encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:8], "big") % (2**31-1))
    
    c = rng.randn(U.shape[1]).astype(np.float32)
    c /= (np.linalg.norm(c) + 1e-12)
    
    b = (U @ torch.from_numpy(c).to(U.device)).float() # (D,)
    return F.normalize(b, p=2, dim=0)

def build_beacon_bank(U: torch.Tensor, secret: str, M: int) -> List[torch.Tensor]:
    return [make_beacon_from_bank(U, secret, m) for m in range(M)]

def assign_beacon_idx(secret: str, gid: str, M: int) -> int:
    mac = hmac.new(secret.encode("utf-8"), gid.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(mac[:4], "big") % M

# =======================
# μ (dataset mean) loader
# =======================
def load_mu_from_embeds(embeds_path: str, device: str) -> torch.Tensor:
    arr = np.load(os.path.expanduser(embeds_path), mmap_mode="r").astype("float32") # (N,D)
    
    # row-wise L2 normalize → mean → normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    
    mu = arr.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-12)
    
    return torch.from_numpy(mu).to(device)

# =======================
# Target Set Selection S
# =======================
def hmac_pick(secret_key: str, msg: str, threshold: float) -> bool:
    mac = hmac.new(secret_key.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).digest()
    val = int.from_bytes(mac[:8], "big") / float(2**64)
    return val < threshold

def load_index(index_parquet: Path) -> pd.DataFrame:
    df = pd.read_parquet(index_parquet)
    df.columns = [str(c) for c in df.columns]
    return df

def build_id_to_path(df: pd.DataFrame, images_root: Path) -> Dict[str, str]:
    gid_col = None
    for c in ["global_id", "gid", "id_str", "id", "key", "index_id", "row_str"]:
        if c in df.columns: gid_col = c; break
    
    path_col = None
    for c in ["path", "filepath", "file", "relpath", "image_path"]:
        if c in df.columns: path_col = c; break
    
    id2path = {}
    
    if gid_col and path_col:
        for _, r in df.iterrows():
            g = str(r[gid_col]); p = str(r[path_col])
            if not os.path.isabs(p): p = str(images_root / p)
            id2path[g] = p
        return id2path

    ds_col = None
    for c in ["dataset","source","ds"]:
        if c in df.columns: ds_col=c; break
    
    file_col = None
    for c in ["filename","file","image","img","name"]:
        if c in df.columns: file_col=c; break
        
    if ds_col and file_col and gid_col:
        for _, r in df.iterrows():
            g = str(r[gid_col]); ds = str(r[ds_col]).lower(); fn = str(r[file_col])
            guesses = [images_root/ ds / "images" / fn, images_root/ ds / fn, images_root/ fn]
            for p in guesses:
                if p.exists():
                    id2path[g] = str(p); break
        return id2path

    return id2path

def select_targets(
    ids_json: Path,
    index_parquet: Path,
    images_root: Path,
    secret_seed: str,
    target_frac: float,
    dataset_filter: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    ids = json.load(open(ids_json))
    ids = [str(x) for x in ids]
    
    df = load_index(index_parquet)
    id2path = build_id_to_path(df, images_root)
    
    dataset_filter = set([d.strip().lower() for d in dataset_filter]) if dataset_filter else None
    
    ds_col = None
    for c in ["dataset","source","ds"]:
        if c in df.columns: ds_col=c; break

    ds_lookup = {}
    if ds_col:
        key_col = None
        for c in ["global_id","gid","id_str","id","key","index_id","row_str"]:
            if c in df.columns: key_col=c; break
        if key_col:
            for _, r in df.iterrows():
                ds_lookup[str(r[key_col])] = str(r[ds_col]).lower()

    out = []
    for gid in ids:
        
        if dataset_filter:
            dsv = ds_lookup.get(gid, None)
            if dsv is None or dsv.lower() not in dataset_filter:
                continue
                
        if not hmac_pick(secret_seed, gid, target_frac):
            continue
            
        p = id2path.get(gid, None)
        if p is None:
            if gid.startswith("webqa_"):
                continue
            else:
                # MMQA specific guess
                guess = images_root / "MMQA" / "images" / f"{gid}.jpg"
                if guess.exists():
                    p = str(guess)
        
        if p is not None and os.path.exists(p):
            out.append((gid, p))
            
    return out

# =======================
# Inverse Optimization
# =======================
def pgd_attack_to_target(
    model,
    preprocess,
    encode_image_infer,
    encode_image_grad,
    device,
    *,
    img_paths: List[str],
    v_targets: torch.Tensor,
    eps: float = 4.0,
    step_size: float = 1.0,
    steps: int = 60,
    norm: str = "linf",
    target_cos: float = 0.35,
    tv_lambda: float = 0.0,
    delta_cos_min: float = 0.05,
    random_start: bool = False,
    mu: float = 0.9,
    cosine_decay: bool = False,
    U: Optional[torch.Tensor] = None,
    e0_batch: Optional[torch.Tensor] = None,
    beta_orth: float = 0.0,
    adaptive_target: bool = False,
    adaptive_margin: float = 0.06,
) -> Tuple[List[np.ndarray], Dict]:
    B = len(img_paths)
    pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
    imgs = torch.stack([preprocess(im) for im in pil_imgs]).to(device)
    imgs.requires_grad_(False)
    v_targets = v_targets.to(device)

    # ctx = torch.amp.autocast("cuda") if (device=="cuda" and torch.cuda.is_available()) else nullcontext
    ctx = (lambda: nullcontext()) 
    
    with torch.no_grad():
        e0 = encode_image_infer(imgs)
        cos0 = (e0 * v_targets).sum(dim=1)
        reached = torch.zeros(B, dtype=torch.bool, device=device)

    def tv_loss(t):
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]
        return dx.abs().mean() + dy.abs().mean()

    # scale: pixel→model space (approx)
    px2norm = 1.0/255.0
    try:
        from torchvision import transforms
        if hasattr(preprocess, "transforms"):
            for t in preprocess.transforms:
                if isinstance(t, transforms.Normalize):
                    std_val = torch.tensor(t.std).to(device)
                    px2norm = float(1.0 / (255.0 * torch.max(std_val).item()))
                    break
    except Exception:
        pass

    eff_eps = eps * px2norm
    base_step = step_size * px2norm
    
    logs = {"before_cos": [], "after_cos": [], "used_steps": 0}

    with torch.no_grad():
        e0 = encode_image_infer(imgs)
        cos0 = (e0 * v_targets).sum(dim=1)
        logs["before_cos"] = cos0.detach().cpu().tolist()

        if adaptive_target:
            per_target = torch.minimum(torch.full_like(cos0, target_cos), cos0 + adaptive_margin)
        else:
            per_target = torch.full_like(cos0, target_cos)

    if beta_orth > 0.0:
        assert U is not None and e0_batch is not None, "U and e0_batch required if beta_orth>0"
        U_ = U.to(device)
        e0b = e0_batch.to(device)
        e0U = e0b @ U_
        e0_proj = e0U @ U_.T
        e0_perp = e0b - e0_proj

    delta = torch.zeros_like(imgs)
    if random_start:
        delta.uniform_(-eff_eps, eff_eps)
    delta = delta.clamp(-eff_eps, eff_eps).detach()
    delta.requires_grad_(True)
    velocity = torch.zeros_like(delta)


    for t in range(steps):
        cur_step = base_step
        if cosine_decay:
            prog = (t + 1) / float(steps)
            cur_step = base_step * 0.5 * (1.0 + math.cos(math.pi * prog))

        with ctx():
            imgs_adv = imgs + delta
            z = encode_image_grad(imgs_adv)
            cos = (z * v_targets).sum(dim=1)
            loss = (-cos).mean()
            
            if tv_lambda > 0:
                loss = loss + tv_lambda * tv_loss(delta)
                
            if beta_orth > 0.0:
                zU = z @ U_
                z_proj = zU @ U_.T
                z_perp = z - z_proj
                loss = loss + beta_orth * (z_perp - e0_perp).pow(2).mean()

            loss.backward()

        with torch.no_grad():
            g = delta.grad.detach()

            if norm == "linf":
                g_sign = g.sign()
                if mu > 0:
                    velocity.mul_(mu).add_(g_sign)
                    step_dir = velocity.sign()
                else:
                    step_dir = g_sign
                
                delta = (delta - cur_step * step_dir).clamp(-eff_eps, eff_eps)

            elif norm == "l2":
                g_flat = g.view(g.shape[0], -1)
                g_norm = g_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                g_unit = (g_flat / g_norm).view_as(g)
                
                if mu > 0:
                    velocity.mul_(mu).add_(g_unit)
                    step_vec = velocity
                else:
                    step_vec = g_unit
                    
                delta = (delta - cur_step * step_vec).detach()
                
                d_flat = delta.view(delta.shape[0], -1)
                d_norm = d_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                over = (d_norm > eff_eps)
                d_proj = d_flat * (eff_eps / d_norm)
                
                delta = torch.where(over, d_proj, d_flat).view_as(delta)

        delta = delta.detach()
        delta.requires_grad_(True)

        if (t+1) % 10 == 0:
            with torch.no_grad():
                cos_now = (encode_image_infer(imgs + delta) * v_targets).sum(dim=1).mean().item()
            print(f"[t={t+1}] cos_mean={cos_now:.4f}")

        with torch.no_grad():
            cos_now = (encode_image_infer(imgs + delta) * v_targets).sum(dim=1)
            reached |= (cos_now >= per_target) | ((cos_now - cos0) >= delta_cos_min)
            if reached.all():
                logs["used_steps"] = t + 1
                break

    with torch.no_grad():
        zf = encode_image_infer(imgs + delta)
        cosf = (zf * v_targets).sum(dim=1)
        logs["after_cos"] = cosf.detach().cpu().tolist()
        
    if logs.get("used_steps", 0) == 0:
        logs["used_steps"] = steps

    # approximate un-normalize back to uint8
    mean, std = None, None
    try:
        from torchvision import transforms
        norm_layer = None
        if hasattr(preprocess, "transforms"):
            for t in preprocess.transforms:
                if isinstance(t, transforms.Normalize):
                    norm_layer = t; break
        
        if norm_layer is not None:
            mean = torch.tensor(norm_layer.mean).to(device).view(1,3,1,1)
            std = torch.tensor(norm_layer.std ).to(device).view(1,3,1,1)
    except Exception:
        pass

    imgs_final = imgs + delta
    
    if mean is not None and std is not None:
        x = imgs_final * std + mean
    else:
        x = imgs_final.clamp(0.0, 1.0)
        
    x = (x.clamp(0,1) * 255.0).round().byte().cpu().numpy()
    x = np.transpose(x, (0,2,3,1))

    return [x[i] for i in range(B)], logs

# =======================
# Glue: end-to-end
# =======================
def _seed_everything_from_secret(secret: str):
    digest = hashlib.sha256((secret + "|attack-seed").encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    
    random.seed(seed_int); np.random.seed(seed_int); torch.manual_seed(seed_int)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed_int)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # NOTE: some versions might require torch.use_deterministic_algorithms(False) to avoid exceptions
        # torch.use_deterministic_algorithms(False) 

def run(args):
    _seed_everything_from_secret(args.secret_seed)
    out_dir = Path(args.out_dir)
    (out_dir/"images").mkdir(parents=True, exist_ok=True)
    meta_f = open(out_dir/"injection_meta.jsonl", "w", encoding="utf-8")

    # 1) Load model/preprocessing
    model, preprocess, encode_image_infer, encode_image_grad, D, device = load_clip_model(args.device)

    # 2) Load U & Beacon Bank, (Optional) μ
    U = seeded_orthonormal_subspace(D, args.subspace_dim, args.secret_seed).to(device) # (D,k)
    M = args.beacon_bank
    beacon_bank = build_beacon_bank(U, args.secret_seed, M) # List[Tensor(D,)]
    
    mu = None
    if args.mix_base == "mu":
        mu = load_mu_from_embeds(args.embeds_path, device) # Tensor(D,)
        
    # 3) Select Target Set
    targets = select_targets(
        ids_json=Path(args.ids_json),
        index_parquet=Path(args.index_parquet),
        images_root=Path(args.images_root),
        secret_seed=args.secret_seed,
        target_frac=args.target_frac,
        dataset_filter=[d.strip() for d in args.dataset_filter.split(",")] if args.dataset_filter else None
    )
    print(f"[INFO] Selected |S|={len(targets)}")

    # 4) Batch Processing
    B = args.batch
    saved = 0
    for i in range(0, len(targets), B):
        chunk = targets[i:i+B]
        gids, paths = zip(*chunk)
        
        with torch.no_grad():
            pil = [Image.open(p).convert("RGB") for p in paths]
            x = torch.stack([preprocess(im) for im in pil]).to(device)
            e0 = encode_image_infer(x) # (b,D)
            e0_batch = e0.detach()

        # Create v_targets: gid -> m -> b_m -> v = (1-α)base + α b_m
        v_list = []
        m_list = []
        for j, gid in enumerate(gids):
            m = assign_beacon_idx(args.secret_seed, gid, M)
            b = beacon_bank[m] # Tensor(D,)
            
            base = (mu if args.mix_base == "mu" else e0[j])
            vj = F.normalize((1.0 - args.alpha) * base + args.alpha * b, p=2, dim=0)
            v_list.append(vj)
            m_list.append(int(m))
        
        v_targets = torch.stack(v_list, dim=0) # (b,D)

        # Attack Optimization
        imgs_adv, logs = pgd_attack_to_target(
            model,
            preprocess,
            encode_image_infer,
            encode_image_grad,
            device,
            img_paths=list(paths),
            v_targets=v_targets,
            eps=args.eps,
            step_size=args.step_size,
            steps=args.steps,
            norm=args.norm,
            target_cos=args.target_cos,
            tv_lambda=args.tv_lambda,
            delta_cos_min=args.delta_cos_min,
            random_start=args.random_start,
            mu=args.mu_momentum,
            cosine_decay=args.cosine_decay,
            U=U,
            e0_batch=e0_batch,
            beta_orth=args.beta_orth,
            adaptive_target=args.adaptive_target,
            adaptive_margin=args.adaptive_margin,
        )

        # Save + Log
        for gid, pth, arr, before_cos, after_cos, m in zip(
            gids, paths, imgs_adv, logs["before_cos"], logs["after_cos"], m_list
        ):
            out_path = out_dir/"images"/f"{gid}.png"
            Image.fromarray(arr).save(out_path)
            
            rec = {
                "id": gid,
                "src_path": pth,
                "out_path": str(out_path),
                "before_cos": float(before_cos),
                "after_cos": float(after_cos),
                "steps_used": logs["used_steps"],
                "eps": args.eps,
                "norm": args.norm,
                "alpha": args.alpha,
                "subspace_dim": args.subspace_dim,
                "beacon_bank": M,
                "assigned_beacon": m,
                "mix_base": args.mix_base,
            }
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
        
        print(f"[INFO] batch {i//B}: saved {saved}/{len(targets)}")

    meta_f.close()
    print("[DONE] injection complete.")

# =======================
# CLI
# =======================
def build_parser():
    ap = argparse.ArgumentParser(description="SONAR — watermark injection (bank-aligned)")
    ap.add_argument("--index-parquet", default="~/SONAR/data/index.parquet")
    ap.add_argument("--ids-json", default="~/SONAR/results/ids.json")
    ap.add_argument("--images-root", default="~/SONAR/data")
    ap.add_argument("--out-dir", default="~/SONAR/beacon/out")
    ap.add_argument("--secret-seed", required=True, help="owner secret")
    ap.add_argument("--subspace-dim", type=int, default=16)

    # NEW: bank + mu
    ap.add_argument("--beacon-bank", type=int, default=32, help="size M of beacon bank")
    ap.add_argument("--embeds-path", default="~/SONAR/results/embeds.npy", help="clean embeds.npy for μ (used when --mix-base mu)")
    ap.add_argument("--mix-base", choices=["mu","e0"], default="mu", help="use dataset mean μ (default) or per-image e0 as the base in mixing")
    
    ap.add_argument("--target-frac", type=float, default=0.05, help="fraction of KB to watermark (0~1), via HMAC sampling")
    ap.add_argument("--dataset-filter", default="WebQA,MMQA", help="comma-separated dataset names to include (match index.parquet)")

    # optimization
    ap.add_argument("--eps", type=float, default=4.0, help="PGD budget (model space approx; see Notes)")
    ap.add_argument("--norm", choices=["linf","l2"], default="linf")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--step-size", type=float, default=1.0, help="PGD step (model space approx)")
    ap.add_argument("--alpha", type=float, default=0.75, help="mix ratio toward beacon vector b_m")
    ap.add_argument("--target-cos",type=float, default=0.45, help="early stop when cos >= target")
    ap.add_argument("--tv-lambda", type=float, default=0.0, help="small TV smoothing on δ (e.g., 1e-4)")
    
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta-cos-min", type=float, default=0.05, help="stop if (cos - cos0) >= this")
    ap.add_argument("--random-start", action="store_true", help="start PGD from a random point within the epsilon ball")
    ap.add_argument("--mu-momentum", type=float, default=0.9, help="momentum coefficient (e.g., 0.7~0.95)")
    ap.add_argument("--cosine-decay", action="store_true", help="use cosine decay schedule on step size")
    ap.add_argument("--beta-orth", type=float, default=0.0, help="subspace-preserving regularizer strength (e.g., 0.1~0.2)")
    ap.add_argument("--adaptive-target", action="store_true", help="use per-sample target: min(target_cos, cos0 + adaptive_margin)")
    ap.add_argument("--adaptive-margin", type=float, default=0.06, help="margin used with --adaptive-target")
    
    return ap

if __name__ == "__main__":
    args = build_parser().parse_args()
    # expanduser for paths
    for k in ["index_parquet","ids_json","images_root","out_dir","embeds_path"]:
        v = getattr(args, k)
        setattr(args, k, os.path.expanduser(v))
    run(args)