\
"""
Make a unified index.parquet for SONAR from MMQA and/or WebQA.

- MMQA: scan MMQA/images/ (or use images.jsonl.gz if present)
- WebQA: create virtual paths like webqa://<row_index> referencing imgs.tsv/lineidx
"""

import os, argparse, gzip, json
from pathlib import Path
import pandas as pd

def add_mmqa(rows, mmqa_root: Path):
    img_dir = mmqa_root/"images"
    if not img_dir.exists():
        return rows
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in [".jpg",".jpeg",".png"]:
            rows.append({
                "id": p.stem,
                "dataset": "MMQA",
                "img_path": str(p.resolve()),
                "split": "unknown",
                "topic": None
            })
    return rows

def add_webqa(rows, webqa_root: Path):
    # Build entries based on lineidx length. img_path uses virtual scheme webqa://<row_idx>
    lineidx = webqa_root/"imgs.lineidx"
    if not lineidx.exists():
        return rows
    with lineidx.open("r", encoding="utf-8") as f:
        offs = [int(x.strip()) for x in f if x.strip()]
    for i in range(len(offs)):
        rows.append({
            "id": f"webqa_{i}",
            "dataset": "WebQA",
            "img_path": f"webqa://{i}",
            "split": "unknown",
            "topic": None
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmqa-root", default="~/SONAR/data/MMQA")
    ap.add_argument("--webqa-root", default="~/SONAR/data/WebQA")
    ap.add_argument("--out", default="~/SONAR/data/index.parquet")
    ap.add_argument("--source", default="both", choices=["mmqa","webqa","both"])
    args = ap.parse_args()

    mmqa_root = Path(args.mmqa_root).expanduser()
    webqa_root = Path(args.webqa_root).expanduser()

    rows = []
    if args.source in ["mmqa","both"]:
        rows = add_mmqa(rows, mmqa_root)
    if args.source in ["webqa","both"]:
        rows = add_webqa(rows, webqa_root)

    df = pd.DataFrame(rows)
    outp = Path(args.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    print(f"wrote {len(df)} rows to {outp}")

if __name__ == "__main__":
    main()
