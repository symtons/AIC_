"""
train.py - Image Captioning with CNN-LSTM and Attention

Train an image captioning model on Flickr8k dataset.
Based on "Show, Attend and Tell" architecture.

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

    # Training hyperparameters
    batch_size = 64
    num_workers = 4
    lr = 1e-4
    num_epochs = 10
    grad_clip = 5.0
    
    # Device
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

    with open(captions_file, "r") as f:
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

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate
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
    print("Image Captioning Training")
    print("="*60)
    print(f"Device: {cfg.device}")
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

    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            encoder, decoder, criterion, optimizer, train_loader, epoch
        )
        val_loss = validate(encoder, decoder, criterion, val_loader, epoch)

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

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
    
    evaluate_bleu_on_test(encoder, decoder, vocab, image2caps, test_pairs)

    print("\n✓ Training finished!")
    print(f"Best model saved to: best_model.pth")


if __name__ == "__main__":
    main()
