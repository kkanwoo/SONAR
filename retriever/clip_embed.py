\
import os, json, yaml, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

# OpenCLIP
import open_clip

# SONAR modules
from retriever.webqa_loader import WebQALoader

ImageFile.MAX_DECOMPRESSED_DATA = 1024 * 1024 * 1024  # 1GB

def load_cfg(p):
    with open(Path(p).expanduser(), "r") as f:
        return yaml.safe_load(f)

def is_webqa_path(p: str) -> bool:
    return str(p).startswith("webqa://")

def decode_webqa_token(token: str) -> int:
    # token format: webqa://<row_idx>
    return int(str(token).split("://", 1)[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", default="~/SONAR/data/index.parquet")
    ap.add_argument("--out_emb", default="~/SONAR/results/embeds.npy")
    ap.add_argument("--out_ids", default="~/SONAR/results/ids.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = "cuda" if (cfg.get("device","cuda")=="cuda" and torch.cuda.is_available()) else "cpu"

    # Model
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"], device=device
    )
    model.eval()

    # For text later if needed
    # tokenizer = open_clip.get_tokenizer(cfg["model"])

    # Dataframe
    df = pd.read_parquet(Path(args.index).expanduser())
    if args.limit:
        df = df.head(args.limit)

    # WebQA loader if needed
    webqa = None
    wq_paths = cfg.get("paths", {}).get("webqa", {})
    tsv_path = wq_paths.get("tsv")
    lineidx_path = wq_paths.get("lineidx")
    if (df["img_path"].astype(str).str.startswith("webqa://").any()) and tsv_path and lineidx_path:
        webqa = WebQALoader(tsv_path, lineidx_path)

    embs = []
    ids = []
    errors = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            pid = row["id"]
            pth = row["img_path"]
            try:
                if is_webqa_path(pth):
                    # WebQA fetch
                    idx = decode_webqa_token(pth)
                    _, img = webqa.get_by_index(idx)
                else:
                    img = Image.open(pth).convert("RGB")

                img_t = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(img_t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embs.append(feat.squeeze(0).cpu().numpy())
                ids.append(pid)
            except Exception as e:
                errors.append((pid, str(e)))

    embs = np.stack(embs, axis=0).astype("float32")
    out_emb = Path(args.out_emb).expanduser()
    out_ids = Path(args.out_ids).expanduser()
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, embs)
    with out_ids.open("w") as f:
        json.dump(ids, f)

    if errors:
        logp = Path("~/SONAR/logs/embedding_errors.txt").expanduser()
        logp.parent.mkdir(parents=True, exist_ok=True)
        with logp.open("w") as f:
            for pid, err in errors:
                f.write(f"{pid}\t{err}\n")

    if webqa:
        webqa.close()

    print(f"embeds: {embs.shape}, ids: {len(ids)}")

if __name__ == "__main__":
    main()
