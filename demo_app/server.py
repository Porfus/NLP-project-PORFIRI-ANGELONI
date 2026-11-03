#!/usr/bin/env python3
"""
Interactive Web Application Server for Italian Dialect Identification & Code-Switching Demo.
Serves a minimal, academic UI and live inference APIs for all 3 trained models:
  1. Logistic Regression + TF-IDF (VarDial Baseline)
  2. Baseline BERT XXL (Pure Transformer)
  3. Dialect Feature Fusion BERT XXL (Proposed Neuro-Symbolic Architecture)
"""

import csv
import json
import importlib
import os
import re
import sys
import time
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig, BertModel

from dataset import DIALECT_CLASSES, ID2LABEL, LABEL2ID
from feature_extraction import _WORD_RE, normalize_text

# ---------------------------------------------------------------------------
# Metadata for the 12 Linguistic Varieties
# ---------------------------------------------------------------------------
DIALECT_META = {
    "it": {
        "code": "it",
        "name": "Standard Italian",
        "native": "Italiano",
        "region": "Italy (National / Central)",
        "group": "Italo-Dalmatian",
        "color": "#2563eb",
    },
    "nap": {
        "code": "nap",
        "name": "Neapolitan",
        "native": "Nnapulitano",
        "region": "Campania & Southern Italy",
        "group": "Italo-Dalmatian (Southern)",
        "color": "#ea580c",
    },
    "scn": {
        "code": "scn",
        "name": "Sicilian",
        "native": "Sicilianu",
        "region": "Sicily, Southern Calabria, Salento",
        "group": "Extreme Southern",
        "color": "#dc2626",
    },
    "vec": {
        "code": "vec",
        "name": "Venetian",
        "native": "Vèneto",
        "region": "Veneto & Northeast",
        "group": "Venetian",
        "color": "#0891b2",
    },
    "lmo": {
        "code": "lmo",
        "name": "Lombard",
        "native": "Lombard",
        "region": "Lombardy",
        "group": "Gallo-Italic",
        "color": "#059669",
    },
    "pms": {
        "code": "pms",
        "name": "Piedmontese",
        "native": "Piemontèis",
        "region": "Piedmont",
        "group": "Gallo-Italic",
        "color": "#4f46e5",
    },
    "lij": {
        "code": "lij",
        "name": "Ligurian",
        "native": "Lìgure / Zeneize",
        "region": "Liguria",
        "group": "Gallo-Italic",
        "color": "#0d9488",
    },
    "eml": {
        "code": "eml",
        "name": "Emilian-Romagnol",
        "native": "Emiliàn-Rumagnòl",
        "region": "Emilia-Romagna",
        "group": "Gallo-Italic",
        "color": "#d97706",
    },
    "fur": {
        "code": "fur",
        "name": "Friulian",
        "native": "Furlan",
        "region": "Friuli-Venezia Giulia",
        "group": "Rhaeto-Romance",
        "color": "#7c3aed",
    },
    "lld": {
        "code": "lld",
        "name": "Ladin",
        "native": "Ladin",
        "region": "Trentino-South Tyrol & Dolomites",
        "group": "Rhaeto-Romance",
        "color": "#9333ea",
    },
    "roa_tara": {
        "code": "roa_tara",
        "name": "Tarantino",
        "native": "Tarandíne",
        "region": "Apulia (Taranto)",
        "group": "Upper Southern",
        "color": "#e11d48",
    },
    "sc": {
        "code": "sc",
        "name": "Sardinian",
        "native": "Sardu",
        "region": "Sardinia",
        "group": "Insular Romance",
        "color": "#ca8a04",
    },
}

REGION_NORMALIZATION = {
    "Campania": ("Campania", "nap"),
    "Veneto": ("Veneto", "vec"),
    "Sicily": ("Sicily (Sicilia)", "scn"),
    "Sicilia": ("Sicily (Sicilia)", "scn"),
    "Lombardy": ("Lombardy (Lombardia)", "lmo"),
    "Lombardia": ("Lombardy (Lombardia)", "lmo"),
    "Sardinia": ("Sardinia (Sardegna)", "sc"),
    "Sardegna": ("Sardinia (Sardegna)", "sc"),
    "Emilia Romagna": ("Emilia-Romagna", "eml"),
    "Liguria": ("Liguria", "lij"),
    "Friuli-Venezia Giulia": ("Friuli-Venezia Giulia", "fur"),
    "Piemonte": ("Piedmont (Piemonte)", "pms"),
    "Piedmont": ("Piedmont (Piemonte)", "pms"),
    "Puglia": ("Apulia (Puglia)", "roa_tara"),
    "Apulia": ("Apulia (Puglia)", "roa_tara"),
    "Trentino-South Tyrol": ("Trentino-South Tyrol", "lld"),
    "Trentino-Alto Adige": ("Trentino-South Tyrol", "lld"),
    "Lazio": ("Lazio", "it"),
    "Tuscany": ("Tuscany (Toscana)", "it"),
    "Toscana": ("Tuscany (Toscana)", "it"),
    "Calabria": ("Calabria", "scn"),
    "Marche": ("Marche", "it"),
    "Abruzzo": ("Abruzzo", "nap"),
    "Umbria": ("Umbria", "it"),
    "Basilicata": ("Basilicata", "nap"),
    "Molise": ("Molise", "nap"),
    "Valle d'Aosta": ("Aosta Valley", "pms"),
    "Aosta Valley": ("Aosta Valley", "pms"),
}

PRESET_EXAMPLES = [
    {
        "id": "cs_nap_1",
        "title": "Code-Switching: Italian + Neapolitan",
        "text": "Andrea che non voleva alzarsi e fa teng na mutanda bianca",
        "region": "Campania",
        "expected_dialect": "nap",
        "note": "Mostly Standard Italian syntax with an embedded Neapolitan clause ('teng na mutanda'). Baseline BERT is confused (predicts Lombard at 35.5%), whereas Feature Fusion BERT detects the Neapolitan markers and predicts 'nap' (71.2%).",
    },
    {
        "id": "cs_nap_2",
        "title": "Code-Switching: Commuter Traffic (Campania)",
        "text": "Stamattina stavo andando a lavoro ma nun se po fa niente con sto traffico",
        "region": "Campania",
        "expected_dialect": "nap",
        "note": "Classic intra-sentential switch: standard Italian opening ('Stamattina stavo andando a lavoro') switching into Neapolitan ('nun se po fa niente con sto traffico').",
    },
    {
        "id": "cs_vec_1",
        "title": "Code-Switching: Venetian Clitics & Negation",
        "text": "Tutti a parlare ma qua no se capise niente e no go mia tempo da perder",
        "region": "Veneto",
        "expected_dialect": "vec",
        "note": "Baseline BERT reaches only 53.7% confidence on Venetian due to standard Italian overlap, while Feature Fusion jumps to 99.9% confidence via Venetian markers ('no go', apocope).",
    },
    {
        "id": "cs_scn_1",
        "title": "Code-Switching: Sicilian Phonetic Shibboleths",
        "text": "Ma che cosa dici, chisto è veramente un beddu picciriddu",
        "region": "Sicily (Sicilia)",
        "expected_dialect": "scn",
        "note": "Standard Italian opening followed by retroflex geminate '-dd-' and final '-u' endings ('beddu picciriddu'), activating Sicilian morphological features.",
    },
    {
        "id": "cs_lmo_1",
        "title": "Code-Switching: Milanese / Lombard Post-Verbal Negation",
        "text": "Oggi non avevo voglia di uscire, fa troppo freddo e minga ho capito cosa vuole il scior",
        "region": "Lombardy (Lombardia)",
        "expected_dialect": "lmo",
        "note": "Standard Italian matrix sentence with Lombard lexical shibboleths ('minga', 'scior').",
    },
    {
        "id": "cs_sc_1",
        "title": "Code-Switching: Sardinian Articles & Discourse Markers",
        "text": "Eja ragazzi, stasera andiamo tutti insieme cun sos amigos in piazza",
        "region": "Sardinia (Sardegna)",
        "expected_dialect": "sc",
        "note": "Contains Sardinian affirmative 'eja', preposition 'cun', and sigmatic plural article 'sos'.",
    },
    {
        "id": "ctrl_it_1",
        "title": "Control Sample: Pure Standard Italian",
        "text": "Oggi pomeriggio andiamo a fare una passeggiata in centro città con gli amici",
        "region": "Italy (National / Central)",
        "expected_dialect": "it",
        "note": "Negative control without dialectal markers: all three models accurately classify it as Standard Italian ('it').",
    },
]


# ---------------------------------------------------------------------------
# Offline-Ready Neural Architecture Definitions
# ---------------------------------------------------------------------------
class OfflineBaselineBERT(nn.Module):
    def __init__(self, config, num_classes=12, dropout_prob=0.2):
        super().__init__()
        self.encoder = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        return self.classifier(cls_rep)


class OfflineDialectFeatureFusionBERT(nn.Module):
    def __init__(self, config, num_classes=12, aux_dim=938, aux_proj_dim=128, dropout_prob=0.2):
        super().__init__()
        self.encoder = BertModel(config, add_pooling_layer=True)
        self.cls_dropout = nn.Dropout(dropout_prob)
        self.aux_projector = nn.Sequential(
            nn.Linear(aux_dim, aux_proj_dim),
            nn.BatchNorm1d(aux_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        fused_dim = config.hidden_size + aux_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, aux_features=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = self.cls_dropout(outputs.last_hidden_state[:, 0, :])
        aux_proj = self.aux_projector(aux_features)
        fused = torch.cat([cls_rep, aux_proj], dim=-1)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Global Model Engine
# ---------------------------------------------------------------------------
class DemoModelEngine:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[Engine] Initializing on device: {self.device}")
        t0 = time.time()

        # 1. Load Baseline 1: Logistic Regression
        logreg_path = os.path.join(PROJECT_ROOT, "checkpoints", "baseline_logreg.joblib")
        print(f"[Engine] Loading Logistic Regression from {logreg_path}...")
        logreg_bundle = joblib.load(logreg_path)
        self.logreg_clf = logreg_bundle["model"]
        self.logreg_tfidf = logreg_bundle["tfidf"]
        self.logreg_scaler = logreg_bundle["scaler"]
        self.logreg_classes = list(logreg_bundle["classes"])

        # 2. Load Auxiliary Symbolic Feature Extractor (includes Cased BERT Tokenizer)
        ext_path = os.path.join(PROJECT_ROOT, "checkpoints", "fusion_feature_extractor.joblib")
        print(f"[Engine] Loading Symbolic Feature Extractor (938 features) from {ext_path}...")
        try:
            self.extractor = joblib.load(ext_path)
        except ModuleNotFoundError as exc:
            legacy_module = "transformers.models.bert.tokenization_bert_fast"
            if exc.name != legacy_module:
                raise
            current_module = importlib.import_module(
                "transformers.models.bert.tokenization_bert"
            )
            sys.modules[legacy_module] = current_module
            self.extractor = joblib.load(ext_path)
        tokenizer_path = os.path.join(PROJECT_ROOT, "models", "bert-base-italian-xxl-cased")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.pad_token_id != 0:
            raise ValueError(
                f"Expected tokenizer padding ID 0 for the saved BERT checkpoints, "
                f"got {self.tokenizer.pad_token_id}."
            )
        self.aux_dim = self.extractor.n_features_
        self.feature_names = self.extractor.get_feature_names_out()

        # Checkpoint weights define the exact vocabulary size. Some tokenizer
        # versions report a smaller base vocabulary despite added tokens.
        base_pt = os.path.join(PROJECT_ROOT, "checkpoints", "best_baseline_model.pt")
        fusion_pt = os.path.join(PROJECT_ROOT, "checkpoints", "best_fusion_model.pt")
        sd_base = torch.load(base_pt, map_location="cpu", weights_only=False)
        sd_fusion = torch.load(fusion_pt, map_location="cpu", weights_only=False)
        embedding_key = "encoder.embeddings.word_embeddings.weight"
        base_vocab_size = sd_base[embedding_key].shape[0]
        fusion_vocab_size = sd_fusion[embedding_key].shape[0]
        if base_vocab_size != fusion_vocab_size:
            raise ValueError(
                "Baseline and fusion checkpoints have different vocabulary sizes: "
                f"{base_vocab_size} and {fusion_vocab_size}."
            )

        # 3. Build BertConfig matching the saved model weights.
        bert_config = BertConfig(
            vocab_size=base_vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=self.tokenizer.pad_token_id,
            layer_norm_eps=1e-12,
        )

        # 4. Load Baseline 2: Baseline BERT XXL
        print(f"[Engine] Loading Baseline BERT XXL from {base_pt}...")
        self.model_base = OfflineBaselineBERT(bert_config, num_classes=len(DIALECT_CLASSES))
        self.model_base.load_state_dict(sd_base, strict=False)
        self.model_base.to(self.device).eval()

        # 5. Load Proposed Model: Feature Fusion BERT XXL
        print(f"[Engine] Loading Feature Fusion BERT XXL from {fusion_pt}...")
        self.model_fusion = OfflineDialectFeatureFusionBERT(
            bert_config, num_classes=len(DIALECT_CLASSES), aux_dim=self.aux_dim
        )
        self.model_fusion.load_state_dict(sd_fusion, strict=False)
        self.model_fusion.to(self.device).eval()

        # Precompute projection weight matrix for fast auxiliary inspection
        self.proj_weight = self.model_fusion.aux_projector[0].weight.detach().cpu().numpy()  # (128, 938)

        # 6. Load Social Media Dataset
        self.social_posts = self._load_social_posts()
        self.benchmark_data = self._load_benchmark_data()

        print(f"[Engine] All 3 models and datasets ready in {time.time() - t0:.2f}s!")

    def _load_social_posts(self):
        csv_path = os.path.join(PROJECT_ROOT, "Social_media_posts", "clean.csv")
        posts = []
        if not os.path.exists(csv_path):
            return posts
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_reg = (row.get("region") or "").strip()
                if not raw_reg:
                    continue
                norm_reg, exp_dial = REGION_NORMALIZATION.get(raw_reg, (raw_reg, "it"))
                text = (row.get("text") or "").strip()
                if len(text) < 15:
                    continue
                posts.append({
                    "id": row.get("id", ""),
                    "text": text,
                    "raw_region": raw_reg,
                    "region": norm_reg,
                    "city": (row.get("city") or "").strip(),
                    "expected_dialect": exp_dial,
                })
        return posts

    def _load_benchmark_data(self):
        return {
            "test_samples": 877048,
            "train_samples": 200000,
            "val_samples": 873361,
            "raw_wiki_samples": 4082394,
            "aux_features_total": self.aux_dim,
            "aux_breakdown": {
                "Character N-Grams (2-4, char_wb)": 500,
                "Word-Edge Affixes & Short Words": 300,
                "Dialect Regex Shibboleth Markers": 99,
                "Orthographic & Phonetic Rates": 24,
                "Dialect Aggregate Hit Scores": 11,
                "BERT Tokenizer Fertility Metrics": 4,
            },
            "models": [
                {
                    "name": "Logistic Regression (VarDial Baseline)",
                    "short": "LogReg + TF-IDF",
                    "architecture": "50,000 word/bigram TF-IDF + StandardScaler + SAG Solver (balanced)",
                    "macro_f1": 0.8962,
                    "weighted_f1": 0.9841,
                    "accuracy": 0.9831,
                    "eval_time": "28.1s",
                    "val_macro_f1": 0.8910,
                },
                {
                    "name": "Baseline BERT XXL (Pure Neural)",
                    "short": "Baseline BERT XXL",
                    "architecture": "dbmdz/bert-base-italian-xxl-cased ([CLS] 768-d -> Linear 12)",
                    "macro_f1": 0.9662,
                    "weighted_f1": 0.9954,
                    "accuracy": 0.9953,
                    "eval_time": "153.3s",
                    "val_macro_f1": 0.9873,
                },
                {
                    "name": "Feature Fusion BERT XXL (Proposed)",
                    "short": "Feature Fusion BERT",
                    "architecture": "Neuro-Symbolic: [CLS] 768-d ⊕ Aux Branch (938 -> 128-d) -> 896-d -> 256-d -> 12",
                    "macro_f1": 0.9668,
                    "weighted_f1": 0.9956,
                    "accuracy": 0.9955,
                    "eval_time": "155.3s",
                    "val_macro_f1": 0.9876,
                },
            ],
            "epochs_progression": {
                "epochs": [1, 2, 3],
                "baseline_train_loss": [0.2208, 0.0429, 0.0170],
                "baseline_val_f1": [0.9846, 0.9861, 0.9873],
                "fusion_train_loss": [0.1759, 0.0463, 0.0200],
                "fusion_val_f1": [0.9841, 0.9860, 0.9876],
            },
            "per_dialect_logreg_val": [
                {"code": "it", "precision": 0.9991, "recall": 0.9825, "f1": 0.9907, "support": 783487},
                {"code": "pms", "precision": 0.9524, "recall": 0.9884, "f1": 0.9701, "support": 2328},
                {"code": "sc", "precision": 0.9466, "recall": 0.9897, "f1": 0.9677, "support": 8738},
                {"code": "lmo", "precision": 0.9378, "recall": 0.9836, "f1": 0.9602, "support": 25987},
                {"code": "lld", "precision": 0.9315, "recall": 0.9832, "f1": 0.9566, "support": 4466},
                {"code": "vec", "precision": 0.9291, "recall": 0.9803, "f1": 0.9540, "support": 18767},
                {"code": "fur", "precision": 0.8730, "recall": 0.9726, "f1": 0.9201, "support": 2517},
                {"code": "scn", "precision": 0.7570, "recall": 0.9798, "f1": 0.8541, "support": 11448},
                {"code": "lij", "precision": 0.7633, "recall": 0.9549, "f1": 0.8484, "support": 6100},
                {"code": "nap", "precision": 0.7080, "recall": 0.9645, "f1": 0.8166, "support": 3804},
                {"code": "roa_tara", "precision": 0.5972, "recall": 0.9560, "f1": 0.7352, "support": 3956},
                {"code": "eml", "precision": 0.5805, "recall": 0.9410, "f1": 0.7180, "support": 1763},
            ],
            "marker_coverage": {
                "sardo": 49.0,
                "friulano": 38.9,
                "ladino": 33.1,
                "napoletano": 30.9,
                "tarantino": 27.2,
                "siciliano": 26.3,
                "piemontese": 18.2,
                "veneto": 13.6,
                "emiliano": 12.9,
                "genovese": 10.3,
                "lombardo": 10.1,
            },
        }

    def _compute_word_attributions(self, model, encoding, aux_tensor, target_class_idx, is_fusion=False):
        """
        Computes fast Input x Gradient token-level attributions with respect to
        BERT word embeddings, then aggregates subword pieces back to words.
        """
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        embed_layer = model.encoder.embeddings.word_embeddings
        raw_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)

        outputs = model.encoder(inputs_embeds=raw_embeds, attention_mask=attention_mask)
        if is_fusion:
            cls_rep = model.cls_dropout(outputs.last_hidden_state[:, 0, :])
            aux_proj = model.aux_projector(aux_tensor.to(self.device))
            fused = torch.cat([cls_rep, aux_proj], dim=-1)
            logits = model.classifier(fused)
        else:
            cls_rep = model.dropout(outputs.last_hidden_state[:, 0, :])
            logits = model.classifier(cls_rep)

        # Contrast target class logit against mean logit (or Italian baseline)
        it_idx = LABEL2ID["it"]
        if target_class_idx != it_idx:
            score = logits[0, target_class_idx] - logits[0, it_idx]
        else:
            score = logits[0, target_class_idx] - logits[0].mean()

        model.zero_grad()
        score.backward()

        grads = raw_embeds.grad[0]  # (seq_len, 768)
        emb = raw_embeds.detach()[0]
        # Signed Input x Gradient attribution along embedding dimension
        token_scores = (grads * emb).sum(dim=-1).cpu().numpy()

        ids = input_ids[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        # Group subwords into words for clean display
        word_items = []
        for tok, sc in zip(tokens, token_scores):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if tok.startswith("##") and word_items:
                word_items[-1]["word"] += tok[2:]
                word_items[-1]["subwords"].append(tok)
                word_items[-1]["score"] += float(sc)
            else:
                word_items.append({
                    "word": tok,
                    "subwords": [tok],
                    "score": float(sc),
                })

        max_abs = max((abs(w["score"]) for w in word_items), default=1e-5)
        if max_abs < 1e-6:
            max_abs = 1.0
        for w in word_items:
            w["norm_score"] = round(float(w["score"] / max_abs), 4)
            w["score"] = round(float(w["score"]), 4)
        return word_items

    def _categorize_feature(self, feat_name):
        if feat_name.startswith("char:"):
            return "Character N-Gram", feat_name[5:]
        if feat_name.startswith("edge:"):
            return "Word-Edge / Affix", feat_name[5:]
        if feat_name.startswith("tok__"):
            return "Tokenizer Fertility", feat_name[5:]
        if "__AGG" in feat_name:
            d = feat_name.split("__")[0].capitalize()
            return "Dialect Marker Aggregate", f"{d} (Total Hits)"
        if "__" in feat_name:
            d, m = feat_name.split("__", 1)
            return "Dialect Shibboleth Regex", f"{d.capitalize()}: {m}"
        return "Orthographic Statistic", feat_name

    def analyze_text(self, text, region=None):
        text = (text or "").strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        # -------------------------------------------------------------------
        # 1. Logistic Regression Inference
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        X_tfidf = self.logreg_tfidf.transform([text])
        X_scaled = self.logreg_scaler.transform(X_tfidf)
        logreg_probs_raw = self.logreg_clf.predict_proba(X_scaled)[0]
        logreg_ms = (time.perf_counter() - t0) * 1000.0

        logreg_probs = {cls: float(p) for cls, p in zip(self.logreg_classes, logreg_probs_raw)}
        logreg_pred = max(logreg_probs, key=logreg_probs.get)

        # -------------------------------------------------------------------
        # 2. Tokenization & Symbolic Feature Extraction (938 features)
        # -------------------------------------------------------------------
        t1 = time.perf_counter()
        encoding = self.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=70,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Baseline BERT Forward
        with torch.no_grad():
            logits_base = self.model_base(input_ids, attention_mask)
            probs_base_arr = F.softmax(logits_base, dim=-1)[0].cpu().numpy()
        base_ms = (time.perf_counter() - t1) * 1000.0

        base_probs = {ID2LABEL[i]: float(probs_base_arr[i]) for i in range(len(DIALECT_CLASSES))}
        base_pred = max(base_probs, key=base_probs.get)

        # -------------------------------------------------------------------
        # 3. Feature Fusion BERT Forward
        # -------------------------------------------------------------------
        t2 = time.perf_counter()
        aux_tensor = self.extractor.transform([text])  # (1, 938)
        with torch.no_grad():
            logits_fusion = self.model_fusion(
                input_ids, attention_mask, aux_features=aux_tensor.to(self.device)
            )
            probs_fusion_arr = F.softmax(logits_fusion, dim=-1)[0].cpu().numpy()
        fusion_ms = (time.perf_counter() - t2) * 1000.0

        fusion_probs = {ID2LABEL[i]: float(probs_fusion_arr[i]) for i in range(len(DIALECT_CLASSES))}
        fusion_pred = max(fusion_probs, key=fusion_probs.get)

        # -------------------------------------------------------------------
        # 4. Tokenizer Fertility & Subword Fragmentation Breakdown
        # -------------------------------------------------------------------
        norm_cased = normalize_text(text, lower=False)
        words = _WORD_RE.findall(norm_cased)
        word_fragmentation = []
        total_pieces = 0
        fragmented_count = 0
        for w in words:
            pieces = self.tokenizer.tokenize(w)
            n_p = max(len(pieces), 1)
            total_pieces += n_p
            if n_p >= 3:
                fragmented_count += 1
            word_fragmentation.append({
                "word": w,
                "pieces": pieces,
                "num_pieces": n_p,
                "is_fragmented": n_p >= 2,
            })
        fertility_rate = round(total_pieces / max(len(words), 1), 3)

        # -------------------------------------------------------------------
        # 5. Auxiliary Branch Feature Inspection (Top-15 of 938 features)
        # -------------------------------------------------------------------
        vals = aux_tensor[0].cpu().numpy()
        mean_impact = vals * np.abs(self.proj_weight).mean(axis=0)
        top_indices = np.argsort(np.abs(mean_impact))[::-1][:15]

        top_aux_features = []
        for idx in top_indices:
            raw_name = self.feature_names[idx]
            cat, clean_label = self._categorize_feature(raw_name)
            top_aux_features.append({
                "raw_name": raw_name,
                "category": cat,
                "label": clean_label,
                "z_score": round(float(vals[idx]), 3),
                "projected_impact": round(float(mean_impact[idx]), 4),
            })

        # Also extract active Dialect Regex Markers specifically (even if not in top 15 TF-IDF)
        active_markers = []
        for idx, raw_name in enumerate(self.feature_names):
            if "__" in raw_name and not raw_name.startswith("tok__") and vals[idx] > 0.5:
                cat, clean_label = self._categorize_feature(raw_name)
                active_markers.append({
                    "raw_name": raw_name,
                    "category": cat,
                    "label": clean_label,
                    "z_score": round(float(vals[idx]), 2),
                    "projected_impact": round(float(mean_impact[idx]), 4),
                })
        active_markers.sort(key=lambda x: abs(x["projected_impact"]), reverse=True)

        # -------------------------------------------------------------------
        # 6. Token-Level Attribution (Baseline BERT vs Feature Fusion BERT)
        # -------------------------------------------------------------------
        target_cls_idx = LABEL2ID[fusion_pred]
        attr_fusion = self._compute_word_attributions(
            self.model_fusion, encoding, aux_tensor, target_cls_idx, is_fusion=True
        )
        attr_base = self._compute_word_attributions(
            self.model_base, encoding, None, target_cls_idx, is_fusion=False
        )

        def format_model_result(model_key, title, pred_code, probs_dict, latency_ms):
            sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            top_k = [
                {
                    "code": c,
                    "name": DIALECT_META[c]["name"],
                    "native": DIALECT_META[c]["native"],
                    "region": DIALECT_META[c]["region"],
                    "color": DIALECT_META[c]["color"],
                    "prob": round(p * 100, 2),
                }
                for c, p in sorted_probs[:5]
            ]
            return {
                "key": model_key,
                "title": title,
                "predicted": {
                    "code": pred_code,
                    **DIALECT_META[pred_code],
                    "confidence": round(probs_dict[pred_code] * 100, 2),
                },
                "latency_ms": round(latency_ms, 1),
                "top_classes": top_k,
            }

        region_info = None
        if region:
            norm_reg, exp_code = REGION_NORMALIZATION.get(region, (region, None))
            if exp_code and exp_code in DIALECT_META:
                region_info = {
                    "region": norm_reg,
                    "expected_code": exp_code,
                    "expected_name": DIALECT_META[exp_code]["name"],
                    "matches": {
                        "logreg": logreg_pred == exp_code,
                        "baseline_bert": base_pred == exp_code,
                        "fusion_bert": fusion_pred == exp_code,
                    },
                }

        return {
            "text": text,
            "device": str(self.device),
            "region_evaluation": region_info,
            "models": {
                "logreg": format_model_result(
                    "logreg", "Logistic Regression (VarDial)", logreg_pred, logreg_probs, logreg_ms
                ),
                "baseline_bert": format_model_result(
                    "baseline_bert", "Baseline BERT XXL", base_pred, base_probs, base_ms
                ),
                "fusion_bert": format_model_result(
                    "fusion_bert", "Feature Fusion BERT XXL", fusion_pred, fusion_probs, fusion_ms
                ),
            },
            "tokenization": {
                "num_words": len(words),
                "num_subwords": total_pieces,
                "fertility_rate": fertility_rate,
                "fragmented_words_count": fragmented_count,
                "words": word_fragmentation,
            },
            "explainability": {
                "target_dialect": DIALECT_META[fusion_pred],
                "attributions_fusion": attr_fusion,
                "attributions_baseline": attr_base,
                "top_symbolic_features": top_aux_features,
                "active_dialect_markers": active_markers[:8],
            },
        }


ENGINE = None


class DemoRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file_from_project(self, rel_path, content_type="image/png"):
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            self.send_error(404, "File not found")
            return
        with open(full_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/init":
            # Count social posts per normalized region
            region_counts = {}
            for p in ENGINE.social_posts:
                r = p["region"]
                region_counts[r] = region_counts.get(r, 0) + 1
            regions_sorted = [
                {
                    "region": r,
                    "count": c,
                    "expected_dialect": REGION_NORMALIZATION.get(r.split(" (")[0], (r, "it"))[1],
                }
                for r, c in sorted(region_counts.items(), key=lambda x: -x[1])
            ]
            self._send_json({
                "device": str(ENGINE.device),
                "dialects": DIALECT_META,
                "presets": PRESET_EXAMPLES,
                "benchmark": ENGINE.benchmark_data,
                "social_regions": regions_sorted,
                "total_social_posts": len(ENGINE.social_posts),
            })
            return

        if path == "/api/social-posts":
            region_filter = query.get("region", ["all"])[0]
            search_q = query.get("q", [""])[0].strip().lower()
            limit = int(query.get("limit", ["30"])[0])
            offset = int(query.get("offset", ["0"])[0])

            filtered = ENGINE.social_posts
            if region_filter and region_filter != "all":
                filtered = [p for p in filtered if p["region"] == region_filter]
            if search_q:
                filtered = [
                    p for p in filtered
                    if search_q in p["text"].lower() or search_q in p["city"].lower()
                ]

            slice_posts = filtered[offset : offset + limit]
            self._send_json({
                "total": len(filtered),
                "offset": offset,
                "limit": limit,
                "posts": slice_posts,
            })
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw) if raw else {}
                text = payload.get("text", "")
                region = payload.get("region", None)
                res = ENGINE.analyze_text(text, region=region)
                self._send_json(res)
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=400)
            return

        self.send_error(404, "Endpoint not found")


def run_server(port=8000):
    global ENGINE
    ENGINE = DemoModelEngine()
    server = HTTPServer(("127.0.0.1", port), DemoRequestHandler)
    print(f"\n============================================================")
    print(f"  Dialect Identification & Code-Switching Demo Server")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print(f"  Device:    {ENGINE.device} (All 3 models active)")
    print(f"============================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
