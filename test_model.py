"""
test_model.py - Quick test of your trained model
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Import model classes (copy from train.py)
class EncoderCNN(nn.Module):
    """ResNet-50 based encoder"""
    def __init__(self, encoded_image_size=14, encoder_dim=512):
        super().__init__()
        self.enc_image_size = encoded_image_size
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.conv_project = nn.Conv2d(2048, encoder_dim, kernel_size=1, stride=1)
        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, images):
        features = self.cnn(images)
        features = self.adaptive_pool(features)
        features = self.conv_project(features)
        B, D, H, W = features.size()
        features = features.permute(0, 2, 3, 1).view(B, -1, D)
        return features


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
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    """LSTM decoder with attention"""
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim, pad_idx):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
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
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c


class Vocabulary:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi
    def __len__(self):
        return len(self.itos)


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on {device}...")

checkpoint = torch.load('best_model.pth', map_location=device)
vocab = Vocabulary(checkpoint['vocab']['itos'], checkpoint['vocab']['stoi'])
config = checkpoint['config']

encoder = EncoderCNN(encoder_dim=config['encoder_dim']).to(device)
decoder = DecoderWithAttention(
    vocab_size=len(vocab),
    embed_dim=config['embed_dim'],
    encoder_dim=config['encoder_dim'],
    decoder_dim=config['decoder_dim'],
    attention_dim=config['attention_dim'],
    pad_idx=vocab.stoi['<pad>']
).to(device)

encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

print(f"✓ Model loaded! Vocab size: {len(vocab)}")


@torch.no_grad()
def generate_caption(image_path, max_len=20):
    """Generate caption for an image"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    encoder_out = encoder(img_tensor)
    h, c = decoder.init_hidden_state(encoder_out)
    
    start_idx = vocab.stoi['<start>']
    end_idx = vocab.stoi['<end>']
    
    caption_words = []
    word_idx = start_idx
    
    for _ in range(max_len):
        word = torch.tensor([word_idx]).to(device)
        embeddings = decoder.embedding(word)
        
        context, alpha = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context
        
        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))
        
        preds = decoder.fc(h)
        word_idx = preds.argmax(dim=1).item()
        
        word = vocab.itos[word_idx]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption_words.append(word)
    
    return ' '.join(caption_words)


# Test on a few images
print("\n" + "="*60)
print("TESTING MODEL - Generating Captions")
print("="*60)

test_images = [
    'data/Flickr8k/Images/1000268201_693b08cb0e.jpg',
    'data/Flickr8k/Images/1001773457_577c3a7d70.jpg',
    'data/Flickr8k/Images/1002674143_1b742ab4b8.jpg',
]

for img_path in test_images:
    if os.path.exists(img_path):
        caption = generate_caption(img_path)
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Caption: {caption}")
    else:
        print(f"\n✗ Image not found: {img_path}")

print("\n" + "="*60)
print("✓ Test complete!")
print("="*60)