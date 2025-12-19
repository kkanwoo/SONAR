#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
import yaml
import random

SEED = 1234  #

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    @torch.no_grad()
    def encode_images(pil_list):
        xs = torch.stack([preprocess(img) for img in pil_list]).to(device)
        z  = model.encode_image(xs).float()
        z  = F.normalize(z, p=2, dim=1)
        return z.cpu().numpy().astype("float32")

    return encode_images

def load_cfg(p):
    with open(Path(p).expanduser(), "r") as f:
        return yaml.safe_load(f)

def main():
    set_seeds(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="~/SONAR/results/ids.json")
    ap.add_argument("--embeds", default="~/SONAR/results/embeds.npy")
    ap.add_argument("--meta", required=True, help="injection_meta.jsonl")
    ap.add_argument("--images-dir", required=True, help="watermarked images dir")
    ap.add_argument("--out", default="~/SONAR/results/embeds_watermarked.npy")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--config", required=False)
    args = ap.parse_args()

    if args.config:
        cfg = load_cfg(args.config)
        args.model = cfg.get("model", args.model)
        args.pretrained = cfg.get("pretrained", args.pretrained)

    ids_path = Path(args.ids).expanduser()
    emb_path = Path(args.embeds).expanduser()
    meta_path = Path(args.meta).expanduser()
    img_root  = Path(args.images_dir).expanduser()
    out_path  = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = json.load(open(ids_path, "r"))
    id2row = {str(gid): i for i, gid in enumerate(ids)}
    X = np.load(emb_path).astype("float32")
    assert X.shape[0] == len(ids), "embeds, ids mismatch"

    encode_images = load_model(args.model, args.pretrained, args.device)

    replace = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            gid = str(j["id"])
            p = Path(j.get("out_path", ""))
            if not p.exists():
                candidates = [
                    img_root / f"{gid}.png",
                    img_root / f"{gid}.jpg",
                    img_root / f"{gid}.jpeg",
                    img_root / f"{gid}.webp",
                ]
                p = next((c for c in candidates if c.exists()), None)
            if p is not None and p.exists():
                replace[gid] = str(p)

    print(f"[INFO] replace targets = {len(replace)}")

    gids = list(replace.keys())
    B = args.batch
    replaced = 0
    missing = 0

    for s in range(0, len(gids), B):
        chunk_ids = gids[s:s+B]
        pil_list, rows = [], []
        opened_images = []
        for gid in chunk_ids:
            row = id2row.get(gid, None)
            if row is None:
                missing += 1
                continue
            try:
                im = Image.open(replace[gid]).convert("RGB")
                opened_images.append(im)
                pil_list.append(im)
                rows.append(row)
            except Exception as e:
                print(f"[WARN] open fail {gid}: {e}")

        if not pil_list:
            continue

        Z = encode_images(pil_list)  # (b,D)
        for i, row in enumerate(rows):
            X[row] = Z[i]
        replaced += len(rows)

        for im in opened_images:
            try: im.close()
            except: pass

        if (s // B) % 10 == 0:
            print(f"[INFO] encoded {replaced}/{len(replace)}")

    np.save(out_path, X)
    print(f"[DONE] saved → {out_path}  (replaced={replaced}, missing_id={missing}, total_shape={X.shape})")

if __name__ == "__main__":
    main()
