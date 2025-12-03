from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import os
import pickle
import time

nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Check if model exists
MODEL_PATH = "./model/tulu_translation_model.pth"
LOOKUP_PATH = "./model/tulu_lookup.pkl"

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size) 
        self.l2 = torch.nn.Linear(hidden_size, hidden_size) 
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Global translator instance
translator = {
    'model': None,
    'all_words': None,
    'unique_tags': None,
    'lookup_dict': None,
    'model_loaded': False,
    'lookup_loaded': False
}

def load_model():
    # Load lookup dictionary
    if os.path.exists(LOOKUP_PATH):
        try:
            with open(LOOKUP_PATH, 'rb') as f:
                translator['lookup_dict'] = pickle.load(f)
            translator['lookup_loaded'] = True
            print(f"✓ Loaded lookup dictionary: {len(translator['lookup_dict'])} entries")
        except Exception as e:
            print(f"✗ Failed to load lookup dictionary: {e}")
    
    # Load neural model
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            
            translator['all_words'] = checkpoint['all_words']
            translator['unique_tags'] = checkpoint['tags']
            
            # Initialize model
            translator['model'] = NeuralNet(
                checkpoint['input_size'],
                checkpoint['hidden_size'],
                checkpoint['output_size']
            )
            translator['model'].load_state_dict(checkpoint['model_state'])
            translator['model'].eval()
            translator['model_loaded'] = True
            
            print(f"✓ Loaded neural model")
            print(f"  Vocabulary: {len(translator['all_words'])} words")
            print(f"  Translations: {len(translator['unique_tags'])} unique")
            return True
        except Exception as e:
            print(f"✗ Failed to load neural model: {e}")
            return False
    else:
        print(f"✗ Model file not found at: {MODEL_PATH}")
        print("  Run train.py first to train the model")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        text = data.get('text', '').strip()
        method = data.get('method', 'simple')
        
        if not text:
            return jsonify({'error': 'Please enter text to translate'}), 400
        
        # Use lookup dictionary for simple method
        if method == 'simple':
            if translator['lookup_loaded']:
                text_lower = text.lower()
                
                # Exact match
                if text_lower in translator['lookup_dict']:
                    return jsonify({
                        'translation': translator['lookup_dict'][text_lower],
                        'confidence': '100%',
                        'method': 'lookup'
                    })
                
                # Word-by-word translation
                words = text_lower.split()
                translated_words = []
                
                for word in words:
                    if word in translator['lookup_dict']:
                        translated_words.append(translator['lookup_dict'][word])
                    else:
                        # Find similar words
                        similar = [k for k in translator['lookup_dict'].keys() if word in k]
                        if similar:
                            translated_words.append(translator['lookup_dict'][similar[0]])
                        else:
                            translated_words.append(f"[{word}]")
                
                translation = " ".join(translated_words)
                return jsonify({
                    'translation': translation,
                    'confidence': '100%',
                    'method': 'lookup'
                })
            else:
                return jsonify({'error': 'Lookup dictionary not loaded'}), 500
        
        # Use neural model
        elif method == 'model':
            if translator['model_loaded']:
                try:
                    tokens = tokenize(text.lower())
                    bag = bag_of_words(tokens, translator['all_words'])
                    bag = bag.reshape(1, -1)
                    bag_tensor = torch.from_numpy(bag).float()
                    
                    with torch.no_grad():
                        output = translator['model'](bag_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        _, predicted_idx = torch.max(output, 1)
                        
                        predicted_tag = translator['unique_tags'][predicted_idx.item()]
                        confidence = probabilities[0][predicted_idx.item()].item()
                    
                    return jsonify({
                        'translation': predicted_tag,
                        'confidence': f"{confidence:.1%}",
                        'method': 'neural_model'
                    })
                except Exception as e:
                    return jsonify({'error': f'Model error: {str(e)}'}), 500
            else:
                return jsonify({'error': 'Neural model not loaded'}), 500
        
        return jsonify({'error': 'Invalid method'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Returns system statistics for the dashboard"""
    try:
        # Get categories from dataset
        categories = {}
        dataset_path = "./dataset/"
        
        if os.path.exists(dataset_path):
            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            for file in json_files:
                category = file.replace('.json', '')
                file_path = os.path.join(dataset_path, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            categories[category] = len(data)
                        elif isinstance(data, dict):
                            if 'intents' in data:
                                categories[category] = len(data['intents'])
                            else:
                                categories[category] = 1
                        else:
                            categories[category] = 1
                except:
                    categories[category] = 0
        
        # Get vocabulary and translation stats
        vocab_size = len(translator['all_words']) if translator['all_words'] else 0
        translation_count = len(translator['unique_tags']) if translator['unique_tags'] else 0
        lookup_size = len(translator['lookup_dict']) if translator['lookup_dict'] else 0
        
        # Use lookup size if available, otherwise use translation count
        translations_display = lookup_size if lookup_size > 0 else translation_count
        
        stats = {
            'vocab_size': vocab_size,
            'total_translations': translations_display,
            'lookup_size': lookup_size,
            'categories': categories,
            'model_loaded': translator['model_loaded'],
            'lookup_loaded': translator['lookup_loaded']
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Returns categories for the categories section (different from stats)"""
    try:
        categories = {}
        dataset_path = "./dataset/"
        
        if os.path.exists(dataset_path):
            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            for file in json_files:
                category = file.replace('.json', '')
                file_path = os.path.join(dataset_path, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            categories[category] = len(data)
                        elif isinstance(data, dict):
                            if 'intents' in data:
                                categories[category] = len(data['intents'])
                            else:
                                categories[category] = 1
                        else:
                            categories[category] = 1
                except:
                    categories[category] = 0
        
        return jsonify({'categories': categories})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vocabulary', methods=['GET'])
def get_vocabulary():
    """Returns vocabulary details"""
    if not translator['model_loaded'] and not translator['lookup_loaded']:
        if not load_model():
            return jsonify({'error': 'No models loaded'}), 500
    
    try:
        sample_words = []
        if translator['all_words']:
            sample_words = list(translator['all_words'][:20])
        
        return jsonify({
            'total_words': len(translator['all_words']) if translator['all_words'] else 0,
            'total_translations': len(translator['unique_tags']) if translator['unique_tags'] else 0,
            'sample_words': sample_words
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Returns example phrases"""
    examples = [
        "Tiger", "Monkey", "Elephant", "Lion", "Dog", "Cat",
        "Father", "Mother", "Brother", "Sister", "Grandfather",
        "Good morning", "Good night", "Thank you", "Hello", "How are you",
        "Water", "Food", "Rice", "Fish", "Meat",
        "House", "Car", "Tree", "Sun", "Moon"
    ]
    return jsonify({'examples': examples})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': translator['model_loaded'],
        'lookup_loaded': translator['lookup_loaded'],
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("TULU TRANSLATOR SERVER")
    print("=" * 60)
    
    # Load models on startup
    print("\nLoading models...")
    load_model()
    
    print("\n" + "=" * 60)
    print("SERVER ENDPOINTS:")
    print("=" * 60)
    print("GET  /              - Web interface")
    print("POST /translate     - Translate text")
    print("GET  /stats         - System statistics")
    print("GET  /categories    - Dataset categories")
    print("GET  /vocabulary    - Vocabulary details")
    print("GET  /examples      - Example phrases")
    print("GET  /health        - Health check")
    print("=" * 60)
    
    print("\nStarting server...")
    print("👉 Open: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)