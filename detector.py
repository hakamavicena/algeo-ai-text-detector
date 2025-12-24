
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class GeometricAIDetector:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):

        print(f"[INFO] Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Thresholds (will be set during training)
        self.tau_variance = None    # τ_var for Total Variance
        self.tau_rank = None         # τ_rank for Effective Rank
        
        # Training statistics
        self.stats = {}
        
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Embedding dimension: {self.embedding_dim}")
    
    
    def preprocess_text(self, text: str) -> List[str]:
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Sentence tokenization
        sentences = nltk.sent_tokenize(text)
        
        # Filter very short sentences (< 3 words)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    def create_embedding_matrix(self, text: str) -> Optional[np.ndarray]:
        # Preprocess
        sentences = self.preprocess_text(text)
        
        # Check minimum length requirement (Section 3.1.5)
        if len(sentences) < 5:
            return None
        
        # Encode sentences to embeddings
        embeddings = self.model.encode(
            sentences, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convert to matrix
        X = np.array(embeddings)  # Shape: (n, d)
        
        return X
    
    
    def compute_total_variance(self, X: np.ndarray) -> float:
        n, d = X.shape
        
        if n < 2:
            return 0.0
        
        try:
            x_bar = np.mean(X, axis=0)
            X_c = X - x_bar
            C = (1 / (n - 1)) * (X_c.T @ X_c)
            total_variance = np.trace(C)
            return max(0.0, float(total_variance))
        except Exception as e:
            print(f"[ERROR] Total variance computation failed: {e}")
            return 0.0

    def compute_effective_rank(self, X: np.ndarray) -> float:
        n, d = X.shape
        
        if n < 2:
            return 0.0
        
        try:
            x_bar = np.mean(X, axis=0)
            X_c = X - x_bar
            
            singular_values = np.linalg.svd(X_c, compute_uv=False)
            sigma = singular_values[singular_values > 1e-10]
            
            if len(sigma) == 0:
                return 0.0
            
            r_eff = (np.sum(sigma) ** 2) / np.sum(sigma ** 2)
            return float(np.clip(r_eff, 1.0, len(sigma)))
            
        except Exception as e:
            print(f"[ERROR] Effective rank computation failed: {e}")
            return 0.0

    def compute_features(self, text: str) -> Optional[Dict[str, float]]:
        try:
            X = self.create_embedding_matrix(text)
            
            if X is None:
                return None
            
            return {
                'total_variance': self.compute_total_variance(X),
                'effective_rank': self.compute_effective_rank(X),
                'n_sentences': X.shape[0],
                'embedding_dim': X.shape[1]
            }
        except Exception as e:
            print(f"[ERROR] Feature computation failed: {e}")
            return None
    
    def train_thresholds(self, 
                        human_texts: List[str], 
                        ai_texts: List[str],
                        verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'='*70}")
            print("THRESHOLD CALIBRATION (Section 3.4.1)")
            print(f"{'='*70}")
            print(f"Human texts: {len(human_texts):,}")
            print(f"AI texts: {len(ai_texts):,}")
        
        # Process human texts
        if verbose:
            print("\n[1/2] Processing human texts...")
        
        human_variance = []
        human_rank = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(human_texts, disable=not verbose)
        except ImportError:
            iterator = human_texts
            if verbose:
                print("(Install tqdm for progress bar: pip install tqdm)")
        
        for text in iterator:
            features = self.compute_features(text)
            if features:
                human_variance.append(features['total_variance'])
                human_rank.append(features['effective_rank'])
        
        # Process AI texts
        if verbose:
            print("\n[2/2] Processing AI texts...")
        
        ai_variance = []
        ai_rank = []
        
        try:
            from tqdm import tqdm
            iterator = tqdm(ai_texts, disable=not verbose)
        except ImportError:
            iterator = ai_texts
        
        for text in iterator:
            features = self.compute_features(text)
            if features:
                ai_variance.append(features['total_variance'])
                ai_rank.append(features['effective_rank'])
        
        # Convert to numpy arrays
        human_variance = np.array(human_variance)
        human_rank = np.array(human_rank)
        ai_variance = np.array(ai_variance)
        ai_rank = np.array(ai_rank)
        
        # Check we have enough samples
        if len(human_variance) < 10 or len(ai_variance) < 10:
            raise ValueError("Need at least 10 valid samples per class")
        
        # Compute class means (μ_H and μ_AI)
        mu_H_var = np.mean(human_variance)
        mu_AI_var = np.mean(ai_variance)
        mu_H_rank = np.mean(human_rank)
        mu_AI_rank = np.mean(ai_rank)
        
        # Compute thresholds (Equation 14)
        # τ = (μ_H + μ_AI) / 2
        self.tau_variance = (mu_H_var + mu_AI_var) / 2
        self.tau_rank = (mu_H_rank + mu_AI_rank) / 2
        
        # Store comprehensive statistics
        self.stats = {
            'threshold_variance': self.tau_variance,
            'threshold_rank': self.tau_rank,
            
            # Human statistics
            'human_variance_mean': mu_H_var,
            'human_variance_std': np.std(human_variance, ddof=1),
            'human_variance_min': np.min(human_variance),
            'human_variance_max': np.max(human_variance),
            'human_rank_mean': mu_H_rank,
            'human_rank_std': np.std(human_rank, ddof=1),
            'human_rank_min': np.min(human_rank),
            'human_rank_max': np.max(human_rank),
            
            # AI statistics
            'ai_variance_mean': mu_AI_var,
            'ai_variance_std': np.std(ai_variance, ddof=1),
            'ai_variance_min': np.min(ai_variance),
            'ai_variance_max': np.max(ai_variance),
            'ai_rank_mean': mu_AI_rank,
            'ai_rank_std': np.std(ai_rank, ddof=1),
            'ai_rank_min': np.min(ai_rank),
            'ai_rank_max': np.max(ai_rank),
            
            # Separation metrics
            'separation_variance': abs(mu_H_var - mu_AI_var),
            'separation_rank': abs(mu_H_rank - mu_AI_rank),
            
            # Sample counts
            'n_human_processed': len(human_variance),
            'n_ai_processed': len(ai_variance),
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("CALIBRATION RESULTS")
            print(f"{'='*70}")
            
            print(f"\n📊 Total Variance (Burstiness):")
            print(f"   Human: μ = {mu_H_var:8.4f}, σ = {self.stats['human_variance_std']:7.4f}")
            print(f"   AI:    μ = {mu_AI_var:8.4f}, σ = {self.stats['ai_variance_std']:7.4f}")
            print(f"   ───────────────────────────────────")
            print(f"   Threshold τ_var = {self.tau_variance:.4f}")
            print(f"   Separation: {self.stats['separation_variance']:.4f}")
            
            print(f"\n📐 Effective Rank (Perplexity Proxy):")
            print(f"   Human: μ = {mu_H_rank:8.4f}, σ = {self.stats['human_rank_std']:7.4f}")
            print(f"   AI:    μ = {mu_AI_rank:8.4f}, σ = {self.stats['ai_rank_std']:7.4f}")
            print(f"   ───────────────────────────────────")
            print(f"   Threshold τ_rank = {self.tau_rank:.4f}")
            print(f"   Separation: {self.stats['separation_rank']:.4f}")
            
            print(f"\n✓ Processed: {self.stats['n_human_processed']:,} human, "
                  f"{self.stats['n_ai_processed']:,} AI texts")
        
        return self.stats
    
    
    def classify_weighted_ensemble(self, 
                                   text: str, 
                                   w_rank: float = 0.70, 
                                   w_var: float = 0.30) -> Tuple[str, Dict]:
        # Check thresholds are trained
        if self.tau_rank is None or self.tau_variance is None:
            raise ValueError(
                "Thresholds not calibrated. Call train_thresholds() first."
            )
        
        # Compute features
        features = self.compute_features(text)
        
        if features is None:
            return 'INSUFFICIENT_DATA', {
                'error': 'Too few sentences (minimum 5 required)',
                'n_sentences': len(self.preprocess_text(text))
            }
        
        var_score = features['total_variance']
        rank_score = features['effective_rank']
        
        # Normalize scores relative to thresholds
        # Score / Threshold:
        #   > 1.0 → Above threshold → Human-like
        #   < 1.0 → Below threshold → AI-like
        norm_rank = rank_score / self.tau_rank
        norm_var = var_score / self.tau_variance
        
        # Weighted ensemble combination
        combined_score = w_rank * norm_rank + w_var * norm_var
        
        # Classification decision
        # combined_score >= 1.0 → More human-like
        # combined_score < 1.0 → More AI-like
        if combined_score >= 1.0:
            label = 'HUMAN'
        else:
            label = 'AI'
        
        # Confidence calculation (Section 3.1.5)
        margin = abs(combined_score - 1.0)
        
        if margin > 0.30:
            confidence = 'HIGH'
        elif margin > 0.15:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Compile detailed results
        details = {
            'label': label,
            'confidence': confidence,
            
            # Raw scores
            'total_variance': var_score,
            'effective_rank': rank_score,
            
            # Thresholds
            'threshold_variance': self.tau_variance,
            'threshold_rank': self.tau_rank,
            
            # Normalized scores
            'normalized_variance': norm_var,
            'normalized_rank': norm_rank,
            
            # Ensemble
            'combined_score': combined_score,
            'weights': {'rank': w_rank, 'variance': w_var},
            'margin': margin,
            
            # Individual votes (for transparency)
            'vote_by_rank': 'HUMAN' if rank_score >= self.tau_rank else 'AI',
            'vote_by_variance': 'HUMAN' if var_score >= self.tau_variance else 'AI',
            
            # Metadata
            'n_sentences': features['n_sentences'],
            'embedding_dim': features['embedding_dim']
        }
        
        return label, details
    
    
    def classify(self, text: str) -> Tuple[str, Dict]:
        return self.classify_weighted_ensemble(text)