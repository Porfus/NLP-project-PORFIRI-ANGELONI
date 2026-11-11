# Neuro-Symbolic Italian Dialect Identification & Code-Switching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Release v1.0.0](https://img.shields.io/badge/Release-v1.0.0-green.svg)](https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/tag/v1.0.0)

**Final Project for Natural Language Processing (NLP)**  
Master's Degree in Computer Science  
**Authors:** Luca Porfiri & Demetrio Angeloni

---

## Abstract & Research Questions

Italian language varieties exhibit extreme dialectal fragmentation, often characterized by phonological erosion, distinct non-standard orthographies, and extensive **intra-sentential code-switching** in informal communication and social media.

Standard Italian pre-trained language models (such as `dbmdz/bert-base-italian-xxl-cased` / DiBERTo) suffer from a severe **Tokenization Penalty**: because dialectal shibboleths and morphological roots are out-of-vocabulary (OOV), standard tokenizers aggressively fragment dialectal words into arbitrary subword pieces, diluting semantic coherence and misdirecting self-attention under code-switching conditions.

To solve this, we propose **DialectFeatureFusionBERT**, a dual-branch **Neuro-Symbolic architecture** that bridges deep contextual transformer representations with an explicit **938-dimensional linguistic and morphological feature extractor** (incorporating character n-grams, word-edge affixes, dialect regex shibboleths, phonetic rates, and tokenizer fertility metrics).

<p align="center">
  <img src="figures/three_models_poster_diagra.svg" alt="Architecture Comparison of the Three Evaluated Models" width="100%">
</p>

---

## Official Benchmark Results (877,048 Test Sentences)

Models were trained on **200,000 temperature-sampled sentences** ($T=2.0$) across 12 linguistic varieties extracted from raw Wikipedia dumps and evaluated on the official unseen **877,048 test split**:

| Model Architecture | Feature Representation | Test Macro F1 | Test Weighted F1 | Test Accuracy | Inference Latency (A100) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(VarDial Baseline)* | 50,000 word/bigram TF-IDF + StandardScaler + SAG | `0.8962` | `0.9841` | `0.9831` | ~1.5 ms / sent |
| **Baseline BERT XXL** | Pure DiBERTo Transformer `[CLS]` (768-d) | `0.9662` | `0.9954` | `0.9953` | ~25 ms / sent |
| **DialectFeatureFusionBERT** *(Proposed)* | **Dual-Branch Neuro-Symbolic Fusion** (768-d + 128-d) | **`0.9668`** | **`0.9956`** | **`0.9955`** | ~27 ms / sent |

### Key Linguistic Varieties (12 Classes)
* Standard Italian (`it`)
* Neapolitan / Upper Southern (`nap`)
* Venetian (`vec`)
* Lombard (`lmo`)
* Sicilian / Extreme Southern (`scn`)
* Sardinian (`sc`)
* Emiliano-Romagnolo (`eml`)
* Piedmontese (`pms`)
* Ligurian (`lij`)
* Friulian (`fur`)
* Tarantino / Apulian (`roa_tara`)
* Ladin / Rhaeto-Romance (`lld`)

---

## Interactive Web Demonstration

We built an interactive, minimal web application designed for live demonstration and pitch presentations. It runs 100% offline with zero cloud dependencies on Apple Silicon MPS, CUDA GPU, or CPU.

### Features
1. **Live Comparison Arena (Tab 01)**: Side-by-side inference across all 3 models with real-time probability distribution bars and curated code-switching presets.
2. **Word-Level Model Attribution (Input × Gradient)**: Differentiable saliency gradients directly computed on input embeddings, highlighting in real-time which words push towards the regional dialect versus Standard Italian.
3. **WordPiece Fragmentation & Fertility Rate Inspector**: Visualizes the *Tokenization Penalty* in real time, pinpointing out-of-vocabulary dialectal words fragmented into subwords.
4. **Explainability & 938 Symbolic Features (Tab 02)**: Real-time projection inspection of the top active auxiliary linguistic features and dialectal regex shibboleths.
5. **Out-of-Domain Social Media Explorer (Tab 03)**: Search and test **2,003 curated geolocated social media posts** across 11 Italian regions, demonstrating how Feature Fusion BERT solves code-switching cases where baseline models fail.
6. **Methodology & Architecture (Tab 04)**: Complete technical reference, confusion matrices, and benchmark summaries.

---

## Quick Start (Run in 1 Click)

You can launch the complete demonstration with a single command. The launch script automatically checks model checkpoints, downloads them if missing, configures the environment, and launches the local web server:

```bash
# 1. Clone repository
git clone https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI.git
cd NLP-project-PORFIRI-ANGELONI

# 2. Run the demonstration script
./run_demo.sh
``` 

Then open your browser at **`http://127.0.0.1:8000`**.

> [!NOTE]
> The script will automatically check if the PyTorch checkpoints exist in `checkpoints/`. If they are not found, it downloads them automatically from [GitHub Releases v1.0.0](https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/tag/v1.0.0).

---

## Model Weights & GitHub Release Assets

The model weights (~850 MB total) are hosted directly on **GitHub Releases v1.0.0** to comply with repository file size limits:

| File | Size | Description | Download Link |
| :--- | :---: | :--- | :--- |
| `best_baseline_model.pt` | ~422 MB | Fine-tuned Baseline DiBERTo XXL Weights | [Download](https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_baseline_model.pt) |
| `best_fusion_model.pt` | ~424 MB | Fine-tuned Neuro-Symbolic Feature Fusion Weights | [Download](https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_fusion_model.pt) |
| `baseline_logreg.joblib` | 7.5 MB | Trained TF-IDF + Logistic Regression Model | Included in repository |
| `fusion_feature_extractor.joblib`| 515 KB | Symbolic Auxiliary Extractor Pipeline | Included in repository |

To download manually, place both `.pt` files into the `checkpoints/` directory:
```bash
mkdir -p checkpoints
curl -L -o checkpoints/best_baseline_model.pt "https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_baseline_model.pt"
curl -L -o checkpoints/best_fusion_model.pt "https://github.com/Porfus/NLP-project-PORFIRI-ANGELONI/releases/download/v1.0.0/best_fusion_model.pt"
```

---

## Repository Structure

```
NLP-project-PORFIRI-ANGELONI/
├── checkpoints/                              # Model checkpoints & benchmark artifacts
│   ├── baseline_logreg.joblib               # Trained Scikit-Learn baseline
│   ├── fusion_feature_extractor.joblib      # 938-feature auxiliary pipeline
│   ├── best_baseline_model.pt               # [Release asset] Baseline BERT weights
│   ├── best_fusion_model.pt                 # [Release asset] Feature Fusion BERT weights
│   ├── benchmark_summary.csv                # Official test split results
│   └── cm_baseline_logreg.png               # Normalized confusion matrix
├── demo_app/                                 # Interactive Web Demonstration
│   ├── server.py                            # Multi-model PyTorch backend & API
│   └── static/
│       ├── index.html                       # 4-Tab clean academic UI
│       ├── style.css                        # Modern responsive layout
│       └── app.js                           # Real-time client & attribution renderer
├── src/                                      # Core library & models
│   ├── dataset.py                           # Dataset loaders & temperature sampling
│   ├── features.py                          # 938-feature symbolic extractor
│   ├── models.py                            # Baseline & FeatureFusion PyTorch modules
│   └── explainability.py                    # Input x Gradient and SHAP wrappers
├── Social_media_posts/                       # Social media code-switching dataset
│   ├── demo_posts_curated.csv               # 2,003 curated posts across 11 target regions
│   └── demo_posts_curated.json              # Pre-computed showcase and consensus tiers
├── executed_interpretability_and_codeswitching.ipynb  # Executed SHAP notebook
├── run_demo.sh                              # 1-Click demonstration launch script
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## Manual Installation & Requirements

If you prefer to set up your own Python environment instead of using `./run_demo.sh`:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start demo server on port 8000
python demo_app/server.py 8000
```

---

## Authors & Acknowledgments

* **Luca Porfiri** — University of Camerino (UNICAM)
* **Demetrio Angeloni** — University of Camerino (UNICAM)

Developed for the **Natural Language Processing** course examination.

