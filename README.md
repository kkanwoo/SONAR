SONARSecret Orthogonal-subspace Nexus for Anchored Retrieval<p align="center"><img src="docs/figures/main_figure.png" alt="SONAR architecture overview" width="900"></p>📂 Dataset PreparationTo reproduce the experiments, please download the required datasets (WebQA, MMQA, VizWiz) and organize them into the data/ directory. The project expects the following directory structure:SONAR/
└── data/
    ├── MMQA/          # MMQA dataset images and metadata
    ├── VizWiz/        # VizWiz dataset images (specifically for 'harmlessness' checks)
    └── WebQA/         # WebQA dataset images and metadata
🏗️ Project StructureThe repository is organized as follows:SONAR/
├── README.md                     # Project documentation and usage guide
├── requirements.txt              # Python dependencies (e.g., faiss, torch, clip)
├── LICENSE                       # License information
│
├── configs/                      # Configuration files
│   └── retriever.yaml            # Configuration for the retriever module
│
├── data/                         # Dataset directory (WebQA, MMQA, VizWiz)
│   ├── MMQA/
│   ├── VizWiz/
│   └── WebQA/
│
├── beacon/                       # Watermark injection module
│   ├── __init__.py
│   └── sonar_watermark.py        # Core logic for subspace watermark injection
│
├── retriever/                    # Retrieval and embedding module
│   ├── __init__.py
│   ├── clip_embed.py             # Script for extracting CLIP embeddings
│   ├── faiss_index.py            # FAISS index training and building
│   ├── make_index.py             # Helper script for index creation
│   ├── make_embeds_watermarked.py # Re-embedding script for watermarked images
│   └── make_image_probes.py      # Generating bank-aware image probes (Optimization)
│
├── eval/                         # Evaluation module
│   ├── __init__.py
│   └── run_eval.py               # Main evaluation script (Retrieval metrics & WSN)
│
└── scripts/                      # Shell scripts for running experiments
    └── run_webmmqa_experiment.sh # Example script for WebQA/MMQA experiments
