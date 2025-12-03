import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

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

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

def load_data(folder_path):
    all_patterns = []
    all_tags = []
    all_categories = []
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        category = json_file.replace('.json', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'en' in item and 'tu' in item:
                            all_patterns.append(item['en'].strip())
                            all_tags.append(item['tu'].strip())
                            all_categories.append(item.get('category', category))
                
                elif isinstance(data, dict):
                    if 'intents' in data:
                        for intent in data['intents']:
                            tag = intent['tag']
                            for pattern in intent.get('patterns', []):
                                all_patterns.append(pattern.strip())
                                all_tags.append(tag.strip())
                                all_categories.append(category)
                    
                    elif 'patterns' in data and 'responses' in data:
                        patterns = data.get('patterns', [])
                        responses = data.get('responses', [])
                        for pattern, response in zip(patterns, responses):
                            all_patterns.append(pattern.strip())
                            all_tags.append(response.strip())
                            all_categories.append(category)
            
            print(f"  Loaded {json_file}")
            
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    return all_patterns, all_tags, all_categories

def main():
    FOLDER_PATH = "./dataset/"
    MODEL_SAVE_PATH = "./model/tulu_translation_model.pth"
    
    os.makedirs("./model", exist_ok=True)
    
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    patterns, tags, categories = load_data(FOLDER_PATH)
    
    print(f"\nTotal samples loaded: {len(patterns)}")
    print(f"Unique English phrases: {len(set(patterns))}")
    print(f"Unique Tulu translations: {len(set(tags))}")
    
    if len(patterns) == 0:
        print("ERROR: No data loaded! Check your JSON files in ./dataset/")
        return
    
    print("\nSample data:")
    for i in range(min(5, len(patterns))):
        print(f"  {i+1}. English: '{patterns[i]}' → Tulu: '{tags[i]}'")
    
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    all_words = []
    ignore_words = ['?', '.', '!', ',', ';', ':', "'", '"', '-', '(', ')', '[', ']']
    
    for pattern in patterns:
        words = tokenize(pattern.lower())
        stemmed_words = [stem(w) for w in words if w not in ignore_words]
        all_words.extend(stemmed_words)
    
    all_words = sorted(set(all_words))
    unique_tags = sorted(set(tags))
    
    print(f"Vocabulary size: {len(all_words)}")
    print(f"Number of unique translations: {len(unique_tags)}")
    
    pattern_to_tag = {}
    for pattern, tag in zip(patterns, tags):
        pattern_lower = pattern.lower()
        if pattern_lower not in pattern_to_tag:
            pattern_to_tag[pattern_lower] = tag
        elif pattern_to_tag[pattern_lower] != tag:
            print(f"  Warning: '{pattern}' has multiple translations")
    
    print(f"Unique English patterns: {len(pattern_to_tag)}")
    
    X_train = []
    y_train = []
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    
    for pattern, tag in pattern_to_tag.items():
        tokens = tokenize(pattern)
        bag = bag_of_words(tokens, all_words)
        X_train.append(bag)
        y_train.append(tag_to_idx[tag])
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    
    print(f"\nTraining data shape: {X_train.shape}")
    
    class TranslationDataset(Dataset):
        def __init__(self, X, y):
            self.n_samples = len(X)
            self.x_data = X
            self.y_data = y
        
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        
        def __len__(self):
            return self.n_samples
    
    dataset = TranslationDataset(X_train, y_train)
    
    if len(dataset) < 10:
        print(f"\nWARNING: Very small dataset ({len(dataset)} samples). Results may not be accurate.")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    batch_size = min(8, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    input_size = len(all_words)
    hidden_size = 64
    output_size = len(unique_tags)
    
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")
    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    num_epochs = 300
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / max(1, len(train_loader))
        train_accuracy = 100 * train_correct / max(1, train_total)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for words, labels in val_loader:
                words = words.to(device)
                labels = labels.to(device)
                
                outputs = model(words)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_accuracy = 100 * val_correct / max(1, val_total)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:4d}/{num_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:6.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:6.2f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    torch.save({
        'model_state': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'all_words': all_words,
        'tags': unique_tags,
        'patterns': list(pattern_to_tag.keys()),
        'translations': list(pattern_to_tag.values()),
        'patterns_original': patterns,
        'translations_original': tags
    }, MODEL_SAVE_PATH)
    
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    model.eval()
    
    test_examples = [
        "Tiger",
        "Monkey", 
        "Elephant",
        "Father",
        "Mother",
        "Good morning",
        "Thank you"
    ]
    
    print("\nTesting translations:\n")
    
    with torch.no_grad():
        for example in test_examples:
            example_lower = example.lower()
            tokens = tokenize(example_lower)
            bag = bag_of_words(tokens, all_words)
            bag = bag.reshape(1, -1)
            bag_tensor = torch.from_numpy(bag).float().to(device)
            
            output = model(bag_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_idx = torch.max(output, 1)
            
            predicted_tag = unique_tags[predicted_idx.item()]
            confidence = probabilities[0][predicted_idx.item()].item()
            
            actual_tag = pattern_to_tag.get(example_lower, "Not in training data")
            
            print(f"English: '{example:15}'")
            print(f"  Predicted Tulu: '{predicted_tag:20}' (Confidence: {confidence:.2%})")
            if actual_tag != "Not in training data":
                print(f"  Actual Tulu:    '{actual_tag:20}'")
                if predicted_tag == actual_tag:
                    print(f"  ✓ CORRECT")
                else:
                    print(f"  ✗ WRONG - Model predicted '{predicted_tag}' instead of '{actual_tag}'")
            else:
                print(f"  ℹ Not in training data")
            print()
    
    print("\n" + "=" * 60)
    print("CREATING SIMPLE LOOKUP DICTIONARY")
    print("=" * 60)
    
    lookup_dict = pattern_to_tag.copy()
    
    lookup_save_path = "./model/tulu_lookup.pkl"
    import pickle
    with open(lookup_save_path, 'wb') as f:
        pickle.dump(lookup_dict, f)
    
    print(f"Lookup dictionary saved to: {lookup_save_path}")
    print(f"Total entries in lookup: {len(lookup_dict)}")
    
    print("\nSample lookup entries:")
    for i, (eng, tulu) in enumerate(list(lookup_dict.items())[:10]):
        print(f"  '{eng:20}' → '{tulu}'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Run the Flask app: python app.py")
    print(f"2. Open http://localhost:5000 in your browser")
    print(f"3. Start translating!")

if __name__ == "__main__":
    main()