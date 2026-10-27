"""
Definizione delle architetture neurali in PyTorch:
1. BaselineBERT: Fine-tuning standard di BERT con pooling [CLS] e classificatore lineare.
2. DialectFeatureFusionBERT: Architettura neuro-simbolica con Feature Fusion. Concatena l'embedding [CLS]
   del Transformer con la proiezione densa delle feature ausiliarie linguistiche e morfologiche.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class BaselineBERT(nn.Module):
    def __init__(self, model_name="dbmdz/bert-base-italian-xxl-cased", num_classes=12, dropout_prob=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token embedding (primo token della sequenza)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep)
        return logits

class DialectFeatureFusionBERT(nn.Module):
    def __init__(
        self,
        model_name="dbmdz/bert-base-italian-xxl-cased",
        num_classes=12,
        aux_dim=501,
        aux_proj_dim=128,
        dropout_prob=0.2
    ):
        super().__init__()
        # Ramo Neurale (Transformer Encodings)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.cls_dropout = nn.Dropout(dropout_prob)

        # Ramo Simbolico (Proiezione feature ausiliarie)
        self.aux_projector = nn.Sequential(
            nn.Linear(aux_dim, aux_proj_dim),
            nn.BatchNorm1d(aux_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Layer di Fusione e Classificatore finale
        fused_dim = hidden_size + aux_proj_dim # es. 768 + 128 = 896
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, aux_features=None, **kwargs):
        # 1. Ramo Neurale
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.cls_dropout(cls_rep)

        # 2. Ramo Ausiliario
        if aux_features is None:
            raise ValueError("aux_features non fornite al modello DialectFeatureFusionBERT!")
        aux_proj = self.aux_projector(aux_features)

        # 3. Concatenazione & Classificazione
        fused = torch.cat([cls_rep, aux_proj], dim=-1)
        logits = self.classifier(fused)
        return logits
