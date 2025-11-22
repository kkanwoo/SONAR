#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, hashlib, random
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import open_clip


# --------------------------
# Utils
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def l2n(x: np.ndarray, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n


def tv_loss(img):
    # img: (B,3,H,W) in [0,1]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())


# --------------------------
# Secret subspace + beacons
# --------------------------
def seeded_U_cpu(D: int, k: int, secret: str) -> torch.Tensor:
    """Deterministic orthonormal basis U (D,k) from secret."""
    rng_t = torch.random.get_rng_state()
    rng_np = np.random.get_state()
    import random as _random
    rng_py = _random.getstate()
    seed = int.from_bytes(hashlib.sha256(secret.encode()).digest()[:8], "big") % (2**31 - 1)
    torch.manual_seed(seed); np.random.seed(seed); _random.seed(seed)
    G = torch.randn(D, k)
    Q, _ = torch.linalg.qr(G, mode="reduced")
    torch.random.set_rng_state(rng_t); np.random.set_state(rng_np); _random.setstate(rng_py)
    return Q[:, :k]


def beacon_vec_bank(U_t: torch.Tensor, secret: str, M: int) -> np.ndarray:
    """Generate M beacon directions in span(U) via PRF seed (D,M)."""
    D, k = U_t.shape
    B = []
    for m in range(M):
        seed_bytes = hashlib.sha256((f"beacon-{secret}-{m}").encode()).digest()
        rs = np.random.RandomState(int.from_bytes(seed_bytes[:8], "big") % (2**31 - 1))
        c = rs.randn(k).astype("float32")
        c /= (np.linalg.norm(c) + 1e-12)
        b = (U_t @ torch.from_numpy(c)).cpu().numpy().astype("float32")
        b /= (np.linalg.norm(b) + 1e-12)
        B.append(b)
    return np.stack(B, axis=1)  # (D,M)


def load_mu_from_embeds(embeds_path: str, chunk: int = 65536) -> np.ndarray:
    arr = np.load(os.path.expanduser(embeds_path), mmap_mode='r')  # (N,D)
    N, D = arr.shape
    acc = np.zeros(D, dtype=np.float64)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X = np.array(arr[s:e], dtype=np.float32, copy=True)
        X = l2n(X, axis=1)
        acc += X.sum(axis=0, dtype=np.float64)
    mu = (acc / N).astype("float32")
    mu = mu / (np.linalg.norm(mu) + 1e-12)
    return mu  # (D,)


# --------------------------
# CLIP image encoder
# --------------------------
def load_clip(model_name: str, pretrained: str, device: str):
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()

    # Preprocess is a torchvision Compose; we separate resize/center-crop/tensor/normalize
    # For optimization we keep an image tensor x in [0,1], then apply normalize before encode.
    norm_layer = None
    to_tensor_layers = []
    resize_crop_layers = []
    if hasattr(preprocess, "transforms"):
        for t in preprocess.transforms:
            if isinstance(t, transforms.Normalize):
                norm_layer = t
            elif isinstance(t, transforms.ToTensor):
                to_tensor_layers.append(t)
            else:
                resize_crop_layers.append(t)
    # We don't need resize/crop since we directly optimize the correct 224x224 size.

    assert norm_layer is not None, "Could not find Normalize layer in CLIP preprocess."

    mean = torch.tensor(norm_layer.mean).view(1,3,1,1)
    std  = torch.tensor(norm_layer.std).view(1,3,1,1)

    @torch.no_grad()
    def encode_image_nograd(x01: torch.Tensor) -> torch.Tensor:
        # x01: (B,3,H,W) in [0,1], float32
        x = (x01 - mean.to(x01.device)) / std.to(x01.device)
        z = model.encode_image(x).float()
        z = F.normalize(z, p=2, dim=1)
        return z

    def encode_image_grad(x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - mean.to(x01.device)) / std.to(x01.device)
        z = model.encode_image(x).float()
        z = F.normalize(z, p=2, dim=1)
        return z

    # infer embedding size
    with torch.no_grad():
        dummy = torch.zeros(1,3,224,224, device=device)
        dim = model.encode_image(dummy).shape[-1]

    return model, encode_image_nograd, encode_image_grad, int(dim), device


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Make image probes aligned to beacon bank")
    ap.add_argument("--secret-seed", required=True)
    ap.add_argument("--subspace-dim", type=int, required=True)
    ap.add_argument("--embeds-path", default="~/SONAR/results/embeds.npy")
    ap.add_argument("--alpha-hat", type=float, default=0.75)
    ap.add_argument("--beacon-bank", type=int, default=32)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda")

    # optimization
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--target-cos", type=float, default=0.70)
    ap.add_argument("--tv-lambda", type=float, default=1e-4)
    ap.add_argument("--init", choices=["zeros","rand"], default="zeros")
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir).expanduser()
    (out_dir/"images").mkdir(parents=True, exist_ok=True)
    meta_f = open(out_dir/"image_probe_meta.jsonl", "w", encoding="utf-8")

    # 1) model
    model, encode_img_nograd, encode_img_grad, D, device = load_clip(args.model, args.pretrained, args.device)

    # 2) make bank and v_m
    mu = load_mu_from_embeds(args.embeds_path)         # (D,)
    U  = seeded_U_cpu(D, args.subspace_dim, args.secret_seed)  # (D,k)
    B  = beacon_vec_bank(U, args.secret_seed, args.beacon_bank)  # (D,M)
    V  = (1.0 - args.alpha_hat) * mu[:,None] + args.alpha_hat * B
    V  = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)   # (D,M)
    Vt = torch.from_numpy(V.T).to(device)  # (M,D)

    # 3) optimize per beacon m
    H, W = 224, 224
    saved = 0
    for m in range(args.beacon_bank):
        # init image in [0,1]
        if args.init == "zeros":
            x = torch.zeros(1,3,H,W, device=device)
        else:
            x = torch.rand(1,3,H,W, device=device)
        x.requires_grad_(True)

        target = Vt[m].unsqueeze(0)  # (1,D)
        opt = torch.optim.Adam([x], lr=args.lr)

        best_cos = -1.0
        for t in range(args.steps):
            z = encode_img_grad(x)               # (1,D), unit
            cos = (z * target).sum(dim=1)        # (1,)
            loss = -cos.mean()
            if args.tv_lambda > 0:
                loss = loss + args.tv_lambda * tv_loss(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                x.clamp_(0,1)
                cval = float(cos.item())
                if cval > best_cos:
                    best_cos = cval
                if cval >= args.target_cos:
                    break

            if (t+1) % 50 == 0:
                print(f"[m={m:03d}] t={t+1}/{args.steps}  cos={cval:.4f}  best={best_cos:.4f}")

        # save png
        with torch.no_grad():
            arr = (x.clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255.0).round().astype(np.uint8)
        out_path = out_dir/"images"/f"probe_m{m:03d}.png"
        Image.fromarray(arr).save(out_path)

        # log
        with torch.no_grad():
            zf = encode_img_nograd(x)
            cosf = float((zf * target).sum().item())
        rec = {
            "m": int(m),
            "path": str(out_path),
            "final_cos": cosf,
            "best_cos": best_cos,
            "steps_used": t+1
        }
        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        saved += 1
        print(f"[DONE] m={m:03d} saved → {out_path}  cos={cosf:.4f}")

    meta_f.close()
    print(f"[DONE] generated {saved} image probes at {out_dir}")


if __name__ == "__main__":
    main()
