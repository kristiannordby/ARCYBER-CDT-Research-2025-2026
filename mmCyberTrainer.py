"""
mmBERT Cyber Security Classifier - Optimized PyTorch Training
Major optimizations:
- Pre-tokenization (tokenize once, not every epoch)
- WeightedRandomSampler (probabilistic ~50/50 balance without data duplication)
- Mixed precision training
- Gradient accumulation for larger effective batch sizes
- Memory efficient - no data duplication

MODIFICATIONS:
- Added language-based evaluation (English vs. non-English)
- Fixed tensor.numpy() RuntimeError with @torch.no_grad()
- Push to hub BEFORE language evaluation to avoid losing trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            roc_auc_score, confusion_matrix, classification_report)

from huggingface_hub import login, HfApi
import numpy as np
import json
import time
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_language_ids(english_path="english_ids.txt", nonenglish_path="nonenglish_ids.txt"):
    """Load English and non-English ID mappings."""
    english_ids = set()
    nonenglish_ids = set()
    
    if os.path.exists(english_path):
        with open(english_path, 'r') as f:
            english_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(english_ids)} English IDs")
    else:
        print(f"Warning: {english_path} not found")
    
    if os.path.exists(nonenglish_path):
        with open(nonenglish_path, 'r') as f:
            nonenglish_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(nonenglish_ids)} non-English IDs")
    else:
        print(f"Warning: {nonenglish_path} not found")
    
    return english_ids, nonenglish_ids


def load_and_preprocess_data(data_path, tokenizer, max_length=512, max_samples=None):
    """Load data and pre-tokenize everything once."""
    print("Loading and pre-tokenizing data...")
    start_time = time.time()
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Store IDs for language-based evaluation
    ids = list(data.keys())
    texts = [data[key]['description'] for key in ids]
    labels = [1 if data[key]['label'] == "true" else 0 for key in ids]
    
    # Limit data if specified
    if max_samples is not None:
        ids = ids[:max_samples]
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Pre-tokenize all texts at once (MUCH faster than one-by-one)
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    print(f"Pre-tokenization completed in {time.time() - start_time:.2f} seconds")
    
    return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels, dtype=torch.long), ids


def create_dataloaders_with_weighted_sampling(input_ids, attention_mask, labels, ids, batch_size=32, 
                                             validation_split=0.15, max_train_samples=None):
    """
    Split data and create dataloaders with WeightedRandomSampler.
    Training set uses weighted sampling to achieve ~50/50 balance without duplicating data.
    Val/test maintain natural distribution.
    """
    # Convert to numpy for sklearn
    indices = np.arange(len(labels))
    labels_np = labels.numpy()
    
    # First split: separate test set (20%)
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels_np
    )
    
    # Second split: separate validation from training
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=validation_split, 
        random_state=42, 
        stratify=labels_np[train_val_indices]
    )
    
    # WEIGHTED SAMPLING FOR TRAINING SET
    print("\n" + "="*70)
    print("SETTING UP WEIGHTED SAMPLING")
    print("="*70)
    
    # Get original training data
    train_input_ids = input_ids[train_indices]
    train_attention_mask = attention_mask[train_indices]
    train_labels = labels[train_indices].numpy()
    
    # Show original distribution
    class_counts = np.bincount(train_labels)
    print(f"Original training distribution:")
    for class_idx, count in enumerate(class_counts):
        print(f"  Class {class_idx}: {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    # Limit training samples if specified (before creating sampler)
    if max_train_samples is not None and len(train_labels) > max_train_samples:
        print(f"\nLimiting training set to {max_train_samples} samples")
        train_input_ids = train_input_ids[:max_train_samples]
        train_attention_mask = train_attention_mask[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
        class_counts = np.bincount(train_labels)
    
    # Calculate weights for balanced sampling (inverse of class frequency)
    weights = 1.0 / class_counts[train_labels]
    
    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True  # Allow sampling with replacement for balance
    )
    
    print(f"\nWeighted sampling configured:")
    print(f"  Total training samples: {len(train_labels)}")
    print(f"  Samples per epoch: {len(weights)}")
    print(f"  Expected balance: ~50/50 (probabilistic)")
    print("="*70 + "\n")
    
    # Create datasets
    train_dataset = TensorDataset(
        train_input_ids,
        train_attention_mask,
        torch.tensor(train_labels, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        input_ids[val_indices],
        attention_mask[val_indices],
        labels[val_indices]
    )
    
    test_dataset = TensorDataset(
        input_ids[test_indices],
        attention_mask[test_indices],
        labels[test_indices]
    )
    
    # Store test IDs for language-based evaluation
    test_ids = [ids[i] for i in test_indices]
    
    print(f"Final dataset sizes:")
    print(f"  Training:   {len(train_dataset)} samples (weighted sampling for ~50/50)")
    print(f"  Validation: {len(val_dataset)} samples (natural distribution)")
    print(f"  Test:       {len(test_dataset)} samples (natural distribution)\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader, test_ids


def build_model(model_name="jhu-clsp/mmBERT-base", device='cuda'):
    """Build and configure the mmBERT binary classification model."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("="*70)
    print("MODEL BUILT")
    print("="*70)
    print(f"Architecture: mmBERT for Binary Classification")
    print(f"Base Model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    return model, tokenizer, optimizer, criterion


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None, accumulation_steps=1):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Get predictions
        with torch.no_grad():
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        total_loss += loss.item() * accumulation_steps
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, auc


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        
        total_loss += loss.item()
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds >= 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, pred_classes),
        'precision': precision_score(all_labels, pred_classes, zero_division=0),
        'recall': recall_score(all_labels, pred_classes, zero_division=0),
        'f1': 2 * precision_score(all_labels, pred_classes, zero_division=0) * recall_score(all_labels, pred_classes, zero_division=0) / 
              (precision_score(all_labels, pred_classes, zero_division=0) + recall_score(all_labels, pred_classes, zero_division=0) + 1e-8),
        'auc': roc_auc_score(all_labels, all_preds)
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, pred_classes)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
    })
    
    return metrics


@torch.no_grad()
def evaluate_by_language(model, test_loader, test_ids, criterion, device, 
                        english_ids, nonenglish_ids):
    """Evaluate model separately on English and non-English test samples."""
    model.eval()
    
    # Collect all predictions and labels with IDs
    all_preds = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc="Evaluating by language"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Separate by language
    english_mask = np.array([tid in english_ids for tid in test_ids])
    nonenglish_mask = np.array([tid in nonenglish_ids for tid in test_ids])
    
    results = {}
    
    # Evaluate English subset
    if english_mask.sum() > 0:
        eng_preds = all_preds[english_mask]
        eng_labels = all_labels[english_mask]
        eng_pred_classes = (eng_preds >= 0.5).astype(int)
        
        results['english'] = {
            'n_samples': int(english_mask.sum()),
            'accuracy': accuracy_score(eng_labels, eng_pred_classes),
            'precision': precision_score(eng_labels, eng_pred_classes, zero_division=0),
            'recall': recall_score(eng_labels, eng_pred_classes, zero_division=0),
            'f1': 2 * precision_score(eng_labels, eng_pred_classes, zero_division=0) * recall_score(eng_labels, eng_pred_classes, zero_division=0) / 
                  (precision_score(eng_labels, eng_pred_classes, zero_division=0) + recall_score(eng_labels, eng_pred_classes, zero_division=0) + 1e-8),
            'auc': roc_auc_score(eng_labels, eng_preds) if len(np.unique(eng_labels)) > 1 else 0.0
        }
        
        cm = confusion_matrix(eng_labels, eng_pred_classes)
        tn, fp, fn, tp = cm.ravel()
        results['english'].update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        })
    
    # Evaluate non-English subset
    if nonenglish_mask.sum() > 0:
        ne_preds = all_preds[nonenglish_mask]
        ne_labels = all_labels[nonenglish_mask]
        ne_pred_classes = (ne_preds >= 0.5).astype(int)
        
        results['nonenglish'] = {
            'n_samples': int(nonenglish_mask.sum()),
            'accuracy': accuracy_score(ne_labels, ne_pred_classes),
            'precision': precision_score(ne_labels, ne_pred_classes, zero_division=0),
            'recall': recall_score(ne_labels, ne_pred_classes, zero_division=0),
            'f1': 2 * precision_score(ne_labels, ne_pred_classes, zero_division=0) * recall_score(ne_labels, ne_pred_classes, zero_division=0) / 
                  (precision_score(ne_labels, ne_pred_classes, zero_division=0) + recall_score(ne_labels, ne_pred_classes, zero_division=0) + 1e-8),
            'auc': roc_auc_score(ne_labels, ne_preds) if len(np.unique(ne_labels)) > 1 else 0.0
        }
        
        cm = confusion_matrix(ne_labels, ne_pred_classes)
        tn, fp, fn, tp = cm.ravel()
        results['nonenglish'].update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        })
    
    return results


def train_model(model, tokenizer, optimizer, criterion, data_path, 
                device='cuda', epochs=50, batch_size=32, patience=10, 
                validation_split=0.15, model_path='mmBERT_cyber_classifier.pt',
                use_amp=True, accumulation_steps=1, max_samples=None, max_train_samples=None):
    """Main training loop with all optimizations."""
    training_start_time = time.time()
    
    # Load and pre-tokenize data
    input_ids, attention_mask, labels, ids = load_and_preprocess_data(data_path, tokenizer, max_samples=max_samples)
    
    # Create dataloaders with weighted sampling
    train_loader, val_loader, test_loader, test_ids = create_dataloaders_with_weighted_sampling(
        input_ids, attention_mask, labels, ids, batch_size, validation_split, max_train_samples
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Training tracking
    best_auc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'epoch_times': []
    }
    
    print("="*70)
    print("TRAINING")
    print("="*70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Accumulation steps: {accumulation_steps} (effective batch: {batch_size * accumulation_steps})")
    print(f"Mixed precision: {use_amp and device.type == 'cuda'}")
    print(f"Early stopping patience: {patience}")
    print("="*70 + "\n")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, accumulation_steps
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['auc'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and checkpointing
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
            }, model_path)
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_time = time.time() - training_start_time
    history['training_time_seconds'] = total_time
    
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"{'='*70}\n")
    
    return model, history, test_loader, test_ids


def write_model_card(output_dir, hub_model_id, metrics, history, config, lang_metrics=None):
    """Generate model card with training results."""
    total_training_time = history.get('training_time_seconds', 0.0)
    
    model_card_content = f"""---
language: en
license: apache-2.0
tags:
- text-classification
- pytorch
- mmbert
- cybersecurity
- binary-classification
metrics:
- accuracy
- f1
- precision
- recall
- auc
model-index:
- name: {hub_model_id}
  results:
  - task:
      type: text-classification
      name: Text Classification
    metrics:
    - name: Accuracy
      type: accuracy
      value: {metrics['accuracy']:.4f}
    - name: F1
      type: f1
      value: {metrics['f1']:.4f}
    - name: Precision
      type: precision
      value: {metrics['precision']:.4f}
    - name: Recall
      type: recall
      value: {metrics['recall']:.4f}
    - name: AUC
      type: auc
      value: {metrics['auc']:.4f}
---

# mmBERT Cybersecurity Classifier (Optimized)

Fine-tuned [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) for cybersecurity content classification.

## Performance

### Overall Test Set

| Metric | Value |
|--------|-------|
| **Accuracy** | {metrics['accuracy']:.4f} |
| **F1 Score** | {metrics['f1']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **AUC** | {metrics['auc']:.4f} |
| **Specificity** | {metrics['specificity']:.4f} |
| **NPV** | {metrics['npv']:.4f} |
| **Training Time** | {total_training_time:.2f}s |

### Confusion Matrix

| Actual/Predicted | Negative | Positive |
|------------------|----------|----------|
| **Negative** | {metrics['tn']} | {metrics['fp']} |
| **Positive** | {metrics['fn']} | {metrics['tp']} |
"""

    if lang_metrics:
        model_card_content += "\n### Performance by Language\n\n"
        
        if 'english' in lang_metrics:
            eng = lang_metrics['english']
            model_card_content += f"""
#### English Articles (n={eng['n_samples']})

| Metric | Value |
|--------|-------|
| **Accuracy** | {eng['accuracy']:.4f} |
| **F1 Score** | {eng['f1']:.4f} |
| **Precision** | {eng['precision']:.4f} |
| **Recall** | {eng['recall']:.4f} |
| **AUC** | {eng['auc']:.4f} |

**Confusion Matrix:**

| Actual/Predicted | Negative | Positive |
|------------------|----------|----------|
| **Negative** | {eng['tn']} | {eng['fp']} |
| **Positive** | {eng['fn']} | {eng['tp']} |
"""
        
        if 'nonenglish' in lang_metrics:
            ne = lang_metrics['nonenglish']
            model_card_content += f"""
#### Non-English Articles (n={ne['n_samples']})

| Metric | Value |
|--------|-------|
| **Accuracy** | {ne['accuracy']:.4f} |
| **F1 Score** | {ne['f1']:.4f} |
| **Precision** | {ne['precision']:.4f} |
| **Recall** | {ne['recall']:.4f} |
| **AUC** | {ne['auc']:.4f} |

**Confusion Matrix:**

| Actual/Predicted | Negative | Positive |
|------------------|----------|----------|
| **Negative** | {ne['tn']} | {ne['fp']} |
| **Positive** | {ne['fn']} | {ne['tp']} |
"""

    model_card_content += f"""
## Training Details

- Pre-tokenization for efficiency
- WeightedRandomSampler (probabilistic ~50/50 class balance)
- No data duplication (memory efficient)
- Mixed precision training
- Early stopping on validation AUC

Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card_content)


def push_to_hub(model, tokenizer, output_dir, hub_model_id, metrics, history):
    """Push model to Hugging Face Hub."""
    print("\n" + "="*70)
    print("PUSHING TO HUB")
    print("="*70)

    with open("hf_token.txt", 'r') as f:
        token = f.read().strip()

    login(token=token)

    commit_msg = f"Optimized - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}"

    model.push_to_hub(hub_model_id, commit_message=commit_msg)
    tokenizer.push_to_hub(hub_model_id)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=os.path.join(output_dir, "README.md"),
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
        token=token,
    )

    print(f"\n✅ Model pushed to https://huggingface.co/{hub_model_id}")


def main():
    """Main training pipeline."""
    config = {
        'model_name': "jhu-clsp/mmBERT-base",
        'data_path': "data/cyber_bert_prepared.json",
        'epochs': 50,
        'batch_size': 128,
        'accumulation_steps': 1,
        'patience': 10,
        'validation_split': 0.15,
        'model_path': './models/cyber_optimized.pt',
        'seed': 42,
        'hub_model_id': 'mmBERT-cyber-classifier-optimized',
        'use_amp': True,
        'lr': 2e-5,
        'max_samples': None,  # Set to limit initial data load (e.g., 200000)
        'max_train_samples': 100000,  # Limit training samples after oversampling
        'english_ids_path': 'english_ids.txt',
        'nonenglish_ids_path': 'nonenglish_ids.txt',
    }

    os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)

    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Load language IDs
    english_ids, nonenglish_ids = load_language_ids(
        config['english_ids_path'], 
        config['nonenglish_ids_path']
    )

    # Build model
    model, tokenizer, optimizer, criterion = build_model(config['model_name'], device)

    # Train
    model, history, test_loader, test_ids = train_model(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        criterion=criterion,
        data_path=config['data_path'],
        device=device,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience'],
        validation_split=config['validation_split'],
        model_path=config['model_path'],
        use_amp=config['use_amp'],
        accumulation_steps=config['accumulation_steps'],
        max_samples=config['max_samples'],
        max_train_samples=config['max_train_samples']
    )

    # Evaluate on full test set first (needed for hub push)
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    metrics = evaluate(model, test_loader, criterion, device)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Push to hub BEFORE language evaluation (so we don't lose the trained model if eval fails)
    output_dir = os.path.dirname(config['model_path']) or "."
    write_model_card(output_dir, config['hub_model_id'], metrics, history, config, lang_metrics=None)
    
    try:
        push_to_hub(model, tokenizer, output_dir, config['hub_model_id'], metrics, history)
        print("\n✅ Model successfully pushed to hub before language evaluation")
    except Exception as e:
        print(f"\n⚠️ Error pushing to hub: {e}")

    # Now do language-based evaluation (if this fails, model is already saved)
    print("\n" + "="*70)
    print("LANGUAGE-BASED EVALUATION")
    print("="*70)
    
    try:
        lang_metrics = evaluate_by_language(
            model, test_loader, test_ids, criterion, device,
            english_ids, nonenglish_ids
        )
        
        if 'english' in lang_metrics:
            print(f"\nEnglish Articles ({lang_metrics['english']['n_samples']} samples):")
            for key, value in lang_metrics['english'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        if 'nonenglish' in lang_metrics:
            print(f"\nNon-English Articles ({lang_metrics['nonenglish']['n_samples']} samples):")
            for key, value in lang_metrics['nonenglish'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
        
        # Update model card with language metrics and push again
        print("\n" + "="*70)
        print("UPDATING MODEL CARD WITH LANGUAGE METRICS")
        print("="*70)
        write_model_card(output_dir, config['hub_model_id'], metrics, history, config, lang_metrics)
        
        try:
            # Just update the README
            with open("hf_token.txt", 'r') as f:
                token = f.read().strip()
            api = HfApi()
            api.upload_file(
                path_or_fileobj=os.path.join(output_dir, "README.md"),
                path_in_repo="README.md",
                repo_id=config['hub_model_id'],
                repo_type="model",
                token=token,
            )
            print("✅ Model card updated with language-specific metrics")
        except Exception as e:
            print(f"⚠️ Could not update model card: {e}")
            
    except Exception as e:
        print(f"\n⚠️ Language evaluation failed: {e}")
        print("Model was already pushed to hub successfully before this error.")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model, history, metrics, lang_metrics


if __name__ == "__main__":
    main()