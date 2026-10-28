import torch
from torch.utils.data import Dataset

DIALECT_CLASSES = sorted([
    'eml', 'fur', 'it', 'lij', 'lld', 'lmo',
    'nap', 'pms', 'roa_tara', 'sc', 'scn', 'vec'
])
LABEL2ID = {d: i for i, d in enumerate(DIALECT_CLASSES)}
ID2LABEL = {i: d for i, d in enumerate(DIALECT_CLASSES)}

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=70, aux_features=None):
        self.texts = texts
        self.labels = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aux_features = aux_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        if self.aux_features is not None:
            item['aux_features'] = self.aux_features[idx]
        return item