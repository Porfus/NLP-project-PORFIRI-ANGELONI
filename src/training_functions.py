import time

import torch
from sklearn.metrics import f1_score, accuracy_score

def fit(model, device, optimizer, scheduler, criterion, scaler, train_loader,
        is_fusion, log_every=200):
    """Trains for one full epoch. Returns (avg_train_loss, elapsed_seconds)."""
    model.train()
    use_amp = device.type == 'cuda'
    total_loss = 0.0
    start = time.time()

    for step, batch in enumerate(train_loader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        kwargs = {}
        if is_fusion:
            kwargs['aux_features'] = batch['aux_features'].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if step % log_every == 0 or step == len(train_loader):
            print(f"  step {step}/{len(train_loader)} | avg loss {total_loss / step:.4f} "
                  f"| {time.time() - start:.0f}s")

    return total_loss / len(train_loader), time.time() - start

def evaluate(model, val_loader, device, criterion, is_fusion=False):
    model.eval()
    use_amp = device.type == 'cuda'
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            kwargs = {}
            if is_fusion:
                kwargs['aux_features'] = batch['aux_features'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, macro_f1, weighted_f1, acc, all_preds, all_labels