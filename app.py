import torch
import pickle
import os
import io
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from detector import GeometricAIDetector
import PyPDF2
import docx
import traceback

class BetterDetector(GeometricAIDetector):
    pass

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == 'pdf':
        text = ""
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    elif ext == 'docx':
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""


MODEL_PATH = 'models/detector_model_v2.pkl'


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    exit(1)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)

try:
    with open(MODEL_PATH, 'rb') as f:
        detector = CPU_Unpickler(f).load()
    
    if hasattr(detector, 'model') and hasattr(detector.model, 'to'):
        detector.model = detector.model.to('cpu')
    
    print("✓ Model loaded (CPU mode)")
    print(f"  Type: {type(detector).__name__}")
    
    
    has_variance = hasattr(detector, 'tau_variance')
    has_var = hasattr(detector, 'tau_var')
    has_rank = hasattr(detector, 'tau_rank')
    
    if has_var and not has_variance:
        print("  Detected Colab model naming (tau_var)")
        detector.tau_variance = detector.tau_var
    elif has_variance and not has_var:
        print("  Detected local model naming (tau_variance)")
        detector.tau_var = detector.tau_variance
    
    if hasattr(detector, 'tau_variance'):
        print(f"  Thresholds:")
        print(f"    - Variance: {detector.tau_variance:.6f}")
        print(f"    - Rank: {detector.tau_rank:.4f}")
    elif hasattr(detector, 'tau_var'):
        print(f"  Thresholds:")
        print(f"    - Variance: {detector.tau_var:.6f}")
        print(f"    - Rank: {detector.tau_rank:.4f}")
    else:
        print("  WARNING: No thresholds found!")
        raise ValueError("Model missing threshold attributes")
    
    if not hasattr(detector, 'classify_weighted_ensemble'):
        print("  WARNING: classify_weighted_ensemble method not found!")
        raise ValueError("Model missing classify_weighted_ensemble method")
    
    print("✓ Model validation passed")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    exit(1)

print("="*80)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    print("\n" + "="*60)
    print("API REQUEST")
    print("="*60)
    
    try:
        text = None
        
        if 'text' in request.form and request.form['text'].strip():
            text = request.form['text']
            print(f"✓ Text: {len(text)} chars")
        
        elif 'file' in request.files:
            file = request.files['file']
            print(f"✓ File: {file.filename}")
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            text = extract_text_from_file(filepath)
            os.remove(filepath)
            
            print(f"✓ Extracted: {len(text)} chars")
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        print("Analyzing...")
        label, details = detector.classify_weighted_ensemble(text)
        
        print(f"✓ Result: {label} ({details.get('confidence', 'N/A')})")
        
        # Build response with safe attribute access
        response = {
            'success': True,
            'classification': label,
            'confidence': details.get('confidence', 'UNKNOWN'),
            'metrics': {
                'effective_rank': round(details.get('effective_rank', 0), 4),
                'total_variance': round(details.get('total_variance', 0), 6),
                'combined_score': round(details.get('combined_score', 0), 4)
            },
            'thresholds': {
                'effective_rank': round(details.get('threshold_rank', 0), 4),
                'total_variance': round(details.get('threshold_variance', 0), 6)
            },
            'analysis': {
                'n_sentences': details.get('n_sentences', 0),
                'margin': round(details.get('margin', 0), 4)
            },
            'votes': {
                'by_rank': details.get('vote_by_rank', 'N/A'),
                'by_variance': details.get('vote_by_variance', 'N/A')
            }
        }
        
        print("="*60 + "\n")
        return jsonify(response)
    
    except Exception as e:
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    rank_threshold = getattr(detector, 'tau_rank', 0)
    var_threshold = getattr(detector, 'tau_variance', None) or getattr(detector, 'tau_var', 0)
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'model_type': type(detector).__name__,
        'device': 'cpu',
        'thresholds': {
            'effective_rank': float(rank_threshold),
            'total_variance': float(var_threshold)
        }
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 STARTING SERVER")
    print("="*80)
    print("Open: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)