# Operate it in Google Colab
# Upload the dataset from https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')
from google.colab import files


uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]
print(f"\n✓ File uploaded: {filename}")

if filename.endswith('.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        # Adjust this based on your pickle structure
        if isinstance(data, dict) and 'csv_filename' in data:
            csv_file = data['csv_filename']
            df = pd.read_csv(csv_file)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
elif filename.endswith('.csv'):
    df = pd.read_csv(filename)
else:
    raise ValueError("Unsupported file format. Please upload .csv or .pkl file")

text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else 'generated' if 'generated' in df.columns else df.columns[1]

# Separate human and AI texts
human_texts = df[df[label_col] == 0][text_col].tolist()
ai_texts = df[df[label_col] == 1][text_col].tolist()

print(f"\nHuman texts: {len(human_texts):,}")
print(f"AI texts: {len(ai_texts):,}")

# Train/test split
h_tr, h_te = train_test_split(human_texts, test_size=0.2, random_state=42)
a_tr, a_te = train_test_split(ai_texts, test_size=0.2, random_state=42)


class BetterDetector:
    def __init__(self):
        print("Loading better model (MPNet)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.tau_var = None
        self.tau_rank = None
        self.stats = {}
        print("✓ MPNet loaded!")

    def embed(self, text):
        try:
            text = ' '.join(str(text).split())
            sents = nltk.sent_tokenize(text)
            sents = [s for s in sents if len(s.split()) >= 3]
            if len(sents) < 5:
                return None
            return np.array(self.model.encode(sents, show_progress_bar=False))
        except:
            return None

    def variance(self, X):
        if X.shape[0] < 2:
            return 0.0
        X_c = X - X.mean(0)
        C = (X_c.T @ X_c) / (X.shape[0] - 1)
        return float(max(0, np.trace(C)))

    def rank(self, X):
        if X.shape[0] < 2:
            return 0.0
        X_c = X - X.mean(0)
        s = np.linalg.svd(X_c, compute_uv=False)
        s = s[s > 1e-10]
        if len(s) == 0:
            return 0.0
        return float(np.clip((s.sum()**2)/(s**2).sum(), 1, len(s)))

    def features(self, text):
        X = self.embed(text)
        if X is None:
            return None
        return {'var': self.variance(X), 'rank': self.rank(X)}

    def train(self, h_texts, a_texts):
        print("\n" + "="*80)
        print("TRAINING WITH MPNET MODEL")
        print("="*80)

        h_var, h_rank = [], []
        print(f"\nProcessing {len(h_texts):,} human texts...")
        for i in tqdm(range(0, len(h_texts), 100)):
            for t in h_texts[i:i+100]:
                f = self.features(t)
                if f:
                    h_var.append(f['var'])
                    h_rank.append(f['rank'])

        a_var, a_rank = [], []
        print(f"\nProcessing {len(a_texts):,} AI texts...")
        for i in tqdm(range(0, len(a_texts), 100)):
            for t in a_texts[i:i+100]:
                f = self.features(t)
                if f:
                    a_var.append(f['var'])
                    a_rank.append(f['rank'])

        h_var = np.array(h_var)
        h_rank = np.array(h_rank)
        a_var = np.array(a_var)
        a_rank = np.array(a_rank)

        self.tau_var = float((h_var.mean() + a_var.mean()) / 2)
        self.tau_rank = float((h_rank.mean() + a_rank.mean()) / 2)

        self.stats = {
            'tau_var': self.tau_var,
            'tau_rank': self.tau_rank,
            'h_var': float(h_var.mean()),
            'a_var': float(a_var.mean()),
            'h_rank': float(h_rank.mean()),
            'a_rank': float(a_rank.mean()),
            'n_h': len(h_var),
            'n_a': len(a_var)
        }

        print("\n" + "="*80)
        print("NEW THRESHOLDS")
        print("="*80)
        print(f"\nVariance: τ = {self.tau_var:.6f}")
        print(f"  Human: {self.stats['h_var']:.6f}")
        print(f"  AI:    {self.stats['a_var']:.6f}")
        print(f"\nRank: τ = {self.tau_rank:.4f}")
        print(f"  Human: {self.stats['h_rank']:.4f}")
        print(f"  AI:    {self.stats['a_rank']:.4f}")

    def classify(self, text):
        f = self.features(text)
        if not f:
            return 'AI'
        r = f['rank'] / self.tau_rank
        v = f['var'] / self.tau_var
        score = 0.7 * r + 0.3 * v
        return 'HUMAN' if score >= 1.0 else 'AI'

det2 = BetterDetector()
det2.train(h_tr, a_tr)

print("\n" + "="*80)
print("TESTING NEW MODEL")
print("="*80)

y_true = [0]*len(h_te) + [1]*len(a_te)
y_pred = []

for t in tqdm(h_te + a_te):
    y_pred.append(1 if det2.classify(t)=='AI' else 0)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nOLD MODEL (MiniLM):")
print(f"  Accuracy: 68.9%")
print(f"\nNEW MODEL (MPNet):")
print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")

# Save jika lebih baik
if acc > 0.689:
    print("\n✓ NEW MODEL IS BETTER! Saving...")

    det2.stats.update({'acc': float(acc), 'prec': float(prec), 'rec': float(rec), 'f1': float(f1)})

    with open('detector_model_v2.pkl', 'wb') as f:
        pickle.dump(det2, f)

    with open('training_stats_v2.json', 'w') as f:
        json.dump(det2.stats, f, indent=2)

    files.download('detector_model_v2.pkl')
    files.download('training_stats_v2.json')

    print("✓ Downloaded improved model!")
else:
    print("\n✗ New model not better. Keep old one.")