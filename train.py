"""
train.py - Image Captioning with CNN-LSTM and Attention

Train an image captioning model on Flickr8k dataset.
Based on "Show, Attend and Tell" architecture.

Optimized for NVIDIA RTX 4080 GPU

Usage:
    python train.py
"""

import os
import re
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


########################################
# Configuration
########################################

class Config:
    """Hyperparameters and paths"""
    # Data paths
    data_dir = "data/Flickr8k"
    images_dir = os.path.join(data_dir, "Images")
    captions_file = os.path.join(data_dir, "captions.txt")

    # Training split ratios
    train_ratio = 0.8
    val_ratio = 0.1  # test = 0.1

    # Vocabulary settings
    min_freq = 5
    max_caption_len = 40

    # Model architecture
    embed_dim = 256
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 256

    # Training hyperparameters - OPTIMIZED FOR RTX 4080
    batch_size = 64        # Increased from 64 (12GB VRAM!)
    num_workers = 8         # Parallel data loading
    lr = 1e-4
    num_epochs = 15
    grad_clip = 5.0
    
    # Device - Auto-detects GPU
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Special tokens
    pad_token = "<pad>"
    start_token = "<start>"
    end_token = "<end>"
    unk_token = "<unk>"

cfg = Config()


########################################
# Plotting Functions
########################################

def plot_training_curves(epochs, train_losses, val_losses, save_path='training_curves.png'):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(left=0)
    
    # Add values on last few points
    for i, (e, tl, vl) in enumerate(zip(epochs[-3:], train_losses[-3:], val_losses[-3:])):
        axes[0].annotate(f'{tl:.3f}', (e, tl), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color='blue')
        axes[0].annotate(f'{vl:.3f}', (e, vl), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Plot 2: Improvement over time
    if len(epochs) > 1:
        train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 
                            for loss in train_losses]
        val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 
                          for loss in val_losses]
        
        axes[1].plot(epochs, train_improvement, 'b-o', label='Train Improvement', 
                    linewidth=2, markersize=6)
        axes[1].plot(epochs, val_improvement, 'r-o', label='Val Improvement', 
                    linewidth=2, markersize=6)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
        axes[1].set_title('Loss Reduction from Epoch 1', fontsize=15, fontweight='bold')
        axes[1].legend(fontsize=12, loc='lower right')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_xlim(left=0)
        
        # Add final improvement values
        final_train_imp = train_improvement[-1]
        final_val_imp = val_improvement[-1]
        axes[1].text(0.95, 0.05, f'Final Train: {final_train_imp:.1f}%\nFinal Val: {final_val_imp:.1f}%',
                    transform=axes[1].transAxes, fontsize=11, verticalalignment='bottom',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {save_path}")


def save_training_summary(epochs, train_losses, val_losses, bleu_score=None):
    """Save training summary to text file"""
    with open('training_summary.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total epochs: {len(epochs)}\n")
        f.write(f"Best train loss: {min(train_losses):.4f} (Epoch {epochs[train_losses.index(min(train_losses))]})\n")
        f.write(f"Best val loss: {min(val_losses):.4f} (Epoch {epochs[val_losses.index(min(val_losses))]})\n")
        f.write(f"Final train loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final val loss: {val_losses[-1]:.4f}\n\n")
        
        if len(epochs) > 1:
            train_change = ((train_losses[-1] - train_losses[0]) / train_losses[0]) * 100
            val_change = ((val_losses[-1] - val_losses[0]) / val_losses[0]) * 100
            f.write(f"Train loss change: {train_change:+.1f}%\n")
            f.write(f"Val loss change: {val_change:+.1f}%\n\n")
        
        if bleu_score is not None:
            f.write(f"Test BLEU-4 Score: {bleu_score:.4f}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"  ✓ Saved summary: training_summary.txt")


########################################
# Utility Functions
########################################

def clean_caption(caption: str) -> str:
    """Clean and normalize caption text"""
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9,.!?']", " ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def tokenize(caption: str):
    """Simple whitespace tokenization"""
    return caption.split()


########################################
# Vocabulary
########################################

class Vocabulary:
    """Build and manage vocabulary"""
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.freqs = Counter()
        self.stoi = {}
        self.itos = []

        self.pad_token = cfg.pad_token
        self.start_token = cfg.start_token
        self.end_token = cfg.end_token
        self.unk_token = cfg.unk_token

    def build(self, all_captions):
        """Build vocabulary from captions"""
        # Count word frequencies
        for cap in all_captions:
            tokens = tokenize(clean_caption(cap))
            self.freqs.update(tokens)

        # Add special tokens first
        self.itos = [
            self.pad_token,
            self.start_token,
            self.end_token,
            self.unk_token,
        ]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

        # Add words that meet frequency threshold
        for word, freq in self.freqs.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def numericalize(self, caption: str):
        """Convert caption to token IDs"""
        tokens = tokenize(clean_caption(caption))
        return [self.stoi.get(tok, self.stoi[self.unk_token]) for tok in tokens]

    def __len__(self):
        return len(self.itos)


########################################
# Data Loading
########################################

def load_captions(captions_file):
    """
    Load captions from file
    
    Returns:
        image2caps: dict mapping image_name to list of captions
        all_pairs: list of (image_name, caption) tuples
    """
    image2caps = {}
    all_pairs = []

    print(f"Loading captions from: {captions_file}")

    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            # Split on first comma
            parts = line.strip().split(",", 1)
            if len(parts) != 2:
                continue
                
            img_id, caption = parts
            img_name = img_id.strip()
            caption = caption.strip()

            # Skip header or non-image rows
            if not img_name.lower().endswith(".jpg"):
                continue

            image2caps.setdefault(img_name, []).append(caption)
            all_pairs.append((img_name, caption))

    print(f"Loaded {len(all_pairs)} captions for {len(image2caps)} images")
    return image2caps, all_pairs


def train_val_test_split(all_pairs, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train/val/test sets"""
    random.seed(seed)
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    
    train_data = all_pairs[:n_train]
    val_data = all_pairs[n_train:n_train + n_val]
    test_data = all_pairs[n_train + n_val:]
    
    return train_data, val_data, test_data


########################################
# Dataset
########################################

class FlickrCaptionDataset(Dataset):
    """Flickr8k dataset loader"""
    def __init__(self, image_caption_pairs, images_dir, vocab: Vocabulary,
                 transform=None, max_len=40):
        self.data = image_caption_pairs
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

        self.start_idx = vocab.stoi[cfg.start_token]
        self.end_idx = vocab.stoi[cfg.end_token]
        self.pad_idx = vocab.stoi[cfg.pad_token]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode caption: <start> tokens <end>
        token_ids = self.vocab.numericalize(caption)
        token_ids = token_ids[: self.max_len - 2]
        caption_ids = [self.start_idx] + token_ids + [self.end_idx]
        length = len(caption_ids)

        return image, torch.tensor(caption_ids, dtype=torch.long), length


class CaptionCollate:
    """Custom collate function for variable-length captions"""
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions, lengths = zip(*batch)
        images = torch.stack(images, dim=0)

        # Pad captions to max length in batch
        max_len = max(lengths)
        padded_captions = torch.full(
            (len(captions), max_len),
            fill_value=self.pad_idx,
            dtype=torch.long,
        )

        for i, cap in enumerate(captions):
            end = cap.shape[0]
            padded_captions[i, :end] = cap

        lengths = torch.tensor(lengths, dtype=torch.long)

        return images, padded_captions, lengths


########################################
# Model: Encoder
########################################

class EncoderCNN(nn.Module):
    """ResNet-50 based encoder"""
    def __init__(self, encoded_image_size=14, encoder_dim=512):
        super().__init__()
        self.enc_image_size = encoded_image_size

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final layers (keep conv features)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Project from 2048 to encoder_dim
        self.conv_project = nn.Conv2d(2048, encoder_dim, kernel_size=1, stride=1)

        self.fine_tune(False)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, L, encoder_dim) where L = enc_size^2
        """
        features = self.cnn(images)              # (B, 2048, H', W')
        features = self.adaptive_pool(features)   # (B, 2048, enc_size, enc_size)
        features = self.conv_project(features)    # (B, encoder_dim, enc_size, enc_size)
        
        # Reshape to sequence
        B, D, H, W = features.size()
        features = features.permute(0, 2, 3, 1).view(B, -1, D)  # (B, L, D)
        
        return features

    def fine_tune(self, fine_tune=True):
        """Freeze/unfreeze encoder layers"""
        for p in self.cnn.parameters():
            p.requires_grad = False
        # Optionally unfreeze later layers
        for c in list(self.cnn.children())[-2:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


########################################
# Model: Attention
########################################

class BahdanauAttention(nn.Module):
    """Additive attention mechanism"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: (B, L, encoder_dim)
            decoder_hidden: (B, decoder_dim)
        Returns:
            context: (B, encoder_dim)
            alpha: (B, L) attention weights
        """
        att1 = self.encoder_att(encoder_out)  # (B, L, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, att_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, L)
        alpha = self.softmax(att)  # (B, L)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


########################################
# Model: Decoder
########################################

class DecoderWithAttention(nn.Module):
    """LSTM decoder with attention"""
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim, pad_idx):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        """Initialize LSTM state from encoder output"""
        mean_encoder = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        Args:
            encoder_out: (B, L, encoder_dim)
            captions: (B, max_len)
            lengths: (B,)
        Returns:
            predictions: (B, max_len-1, vocab_size)
            captions_sorted: sorted captions
            decode_lengths: actual decode lengths
            alphas: attention weights
            sort_idx: sorting indices
        """
        batch_size = encoder_out.size(0)
        L = encoder_out.size(1)

        # Sort by length
        lengths_sorted, sort_idx = lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]

        # Embed captions
        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)

        # Initialize LSTM
        h, c = self.init_hidden_state(encoder_out)

        # Decode lengths
        decode_lengths = (lengths_sorted - 1).tolist()
        max_decode_len = max(decode_lengths)

        # Storage
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, L).to(encoder_out.device)

        # Decode step by step
        for t in range(max_decode_len):
            batch_t = sum([l > t for l in decode_lengths])
            
            # Attention
            context, alpha = self.attention(encoder_out[:batch_t], h[:batch_t])
            gate = self.sigmoid(self.f_beta(h[:batch_t]))
            context = gate * context

            # LSTM step
            lstm_input = torch.cat([embeddings[:batch_t, t, :], context], dim=1)
            h_new, c_new = self.lstm_cell(lstm_input, (h[:batch_t], c[:batch_t]))

            # Update states
            h = torch.cat([h_new, h[batch_t:]], dim=0)
            c = torch.cat([c_new, c[batch_t:]], dim=0)

            # Predict
            preds = self.fc(self.dropout(h_new))
            predictions[:batch_t, t, :] = preds
            alphas[:batch_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sort_idx


########################################
# Training Functions
########################################

def create_dataloaders():
    """Create train/val/test dataloaders"""
    # Load captions
    image2caps, all_pairs = load_captions(cfg.captions_file)
    train_pairs, val_pairs, test_pairs = train_val_test_split(
        all_pairs, cfg.train_ratio, cfg.val_ratio
    )

    # Build vocabulary
    vocab = Vocabulary(min_freq=cfg.min_freq)
    vocab.build([cap for _, cap in train_pairs])
    cfg.pad_token_id = vocab.stoi[cfg.pad_token]

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = FlickrCaptionDataset(
        train_pairs, cfg.images_dir, vocab,
        transform=train_transform, max_len=cfg.max_caption_len
    )
    val_dataset = FlickrCaptionDataset(
        val_pairs, cfg.images_dir, vocab,
        transform=val_transform, max_len=cfg.max_caption_len
    )
    test_dataset = FlickrCaptionDataset(
        test_pairs, cfg.images_dir, vocab,
        transform=val_transform, max_len=cfg.max_caption_len
    )

    collate = CaptionCollate(pad_idx=cfg.pad_token_id)

    # Dataloaders - GPU optimized with pin_memory
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, vocab, image2caps, test_pairs


def train_one_epoch(encoder, decoder, criterion, optimizer, train_loader, epoch):
    """Train for one epoch"""
    encoder.train()
    decoder.train()

    total_loss = 0.0

    for images, captions, lengths in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        images = images.to(cfg.device)
        captions = captions.to(cfg.device)
        lengths = lengths.to(cfg.device)

        optimizer.zero_grad()

        encoder_out = encoder(images)
        scores, caps_sorted, decode_lengths, alphas, sort_idx = decoder(
            encoder_out, captions, lengths
        )

        # Targets: next word after each position
        targets = caps_sorted[:, 1:]

        # Pack predictions and targets
        scores_packed = []
        targets_packed = []
        for i, l in enumerate(decode_lengths):
            scores_packed.append(scores[i, :l, :])
            targets_packed.append(targets[i, :l])

        scores_packed = torch.cat(scores_packed, dim=0)
        targets_packed = torch.cat(targets_packed, dim=0)

        loss = criterion(scores_packed, targets_packed)

        # Attention regularization (doubly stochastic)
        alphas_reg = 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = loss + alphas_reg

        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(encoder, decoder, criterion, val_loader, epoch):
    """Validate the model"""
    encoder.eval()
    decoder.eval()

    total_loss = 0.0

    for images, captions, lengths in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
        images = images.to(cfg.device)
        captions = captions.to(cfg.device)
        lengths = lengths.to(cfg.device)

        encoder_out = encoder(images)
        scores, caps_sorted, decode_lengths, alphas, sort_idx = decoder(
            encoder_out, captions, lengths
        )

        targets = caps_sorted[:, 1:]

        scores_packed = []
        targets_packed = []
        for i, l in enumerate(decode_lengths):
            scores_packed.append(scores[i, :l, :])
            targets_packed.append(targets[i, :l])

        scores_packed = torch.cat(scores_packed, dim=0)
        targets_packed = torch.cat(targets_packed, dim=0)

        loss = criterion(scores_packed, targets_packed)
        alphas_reg = 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        loss = loss + alphas_reg

        total_loss += loss.item()

    return total_loss / len(val_loader)


########################################
# Inference
########################################

@torch.no_grad()
def generate_caption(encoder, decoder, image_tensor, vocab, max_len=20):
    """Generate caption for a single image"""
    encoder.eval()
    decoder.eval()

    image_tensor = image_tensor.to(cfg.device)
    encoder_out = encoder(image_tensor)

    h, c = decoder.init_hidden_state(encoder_out)
    start_idx = vocab.stoi[cfg.start_token]
    end_idx = vocab.stoi[cfg.end_token]

    word_idx = start_idx
    caption_idxs = [start_idx]

    for _ in range(max_len):
        word = torch.tensor([word_idx], dtype=torch.long).to(cfg.device)
        embeddings = decoder.embedding(word)

        context, alpha = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context

        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))
        
        preds = decoder.fc(h)
        _, next_word = preds.max(dim=1)

        word_idx = next_word.item()
        caption_idxs.append(word_idx)
        
        if word_idx == end_idx:
            break

    # Convert to words
    words = []
    for idx in caption_idxs:
        tok = vocab.itos[idx]
        if tok in {cfg.start_token, cfg.end_token, cfg.pad_token}:
            continue
        words.append(tok)

    return " ".join(words)


########################################
# Evaluation
########################################

@torch.no_grad()
def evaluate_bleu_on_test(encoder, decoder, vocab, image2caps, test_pairs):
    """Compute BLEU-4 on test set"""
    encoder.eval()
    decoder.eval()

    # Get unique test images
    test_images = sorted(set(img_name for img_name, _ in test_pairs))

    references = []
    hypotheses = []

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"\nEvaluating BLEU on {len(test_images)} test images...")

    for img_name in tqdm(test_images, desc="BLEU eval"):
        img_path = os.path.join(cfg.images_dir, img_name)

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0)

        # Generate caption
        hyp_str = generate_caption(encoder, decoder, img_tensor, vocab, max_len=20)
        hyp_tokens = hyp_str.split()
        hypotheses.append(hyp_tokens)

        # Get reference captions
        ref_caps = image2caps[img_name]
        ref_tokens = []
        for cap in ref_caps:
            toks = tokenize(clean_caption(cap))
            ref_tokens.append(toks)
        references.append(ref_tokens)

    # BLEU-4 with smoothing
    smoothie = SmoothingFunction().method1
    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    
    print(f"\n{'='*60}")
    print(f"Test BLEU-4 Score: {bleu4:.4f}")
    print('='*60)
    
    return bleu4


########################################
# Main Training Loop
########################################

def main():
    """Main training function"""
    print("="*60)
    print("Image Captioning Training - RTX 4080 Optimized")
    print("="*60)
    print(f"Device: {cfg.device}")
    
    # Display GPU info if available
    if cfg.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Learning rate: {cfg.lr}")
    print("="*60)

    # Create dataloaders
    train_loader, val_loader, test_loader, vocab, image2caps, test_pairs = create_dataloaders()

    # Build models
    encoder = EncoderCNN(encoder_dim=cfg.encoder_dim).to(cfg.device)
    decoder = DecoderWithAttention(
        vocab_size=len(vocab),
        embed_dim=cfg.embed_dim,
        encoder_dim=cfg.encoder_dim,
        decoder_dim=cfg.decoder_dim,
        attention_dim=cfg.attention_dim,
        pad_idx=vocab.stoi[cfg.pad_token],
    ).to(cfg.device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[cfg.pad_token])
    params = list(decoder.parameters()) + list(
        filter(lambda p: p.requires_grad, encoder.parameters())
    )
    optimizer = optim.Adam(params, lr=cfg.lr)

    best_val_loss = float("inf")
    
    # Track training history
    epochs_list = []
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            encoder, decoder, criterion, optimizer, train_loader, epoch
        )
        val_loss = validate(encoder, decoder, criterion, val_loader, epoch)

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Track losses
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Plot progress
        plot_training_curves(epochs_list, train_losses, val_losses)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": {
                    "itos": vocab.itos,
                    "stoi": vocab.stoi,
                },
                "config": {
                    "encoder_dim": cfg.encoder_dim,
                    "decoder_dim": cfg.decoder_dim,
                    "embed_dim": cfg.embed_dim,
                    "attention_dim": cfg.attention_dim,
                },
            }
            
            torch.save(checkpoint, "best_model.pth")
            print("  ✓ Saved best model")

    # Final evaluation
    print("\n" + "="*60)
    print("Training complete! Evaluating on test set...")
    print("="*60)
    
    bleu_score = evaluate_bleu_on_test(encoder, decoder, vocab, image2caps, test_pairs)
    
    # Save final summary
    save_training_summary(epochs_list, train_losses, val_losses, bleu_score)

    print("\n✓ Training finished!")
    print(f"✓ Best model saved to: best_model.pth")
    print(f"✓ Training curves saved to: training_curves.png")
    print(f"✓ Summary saved to: training_summary.txt")


if __name__ == "__main__":
    main()