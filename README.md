# 🌴 Tulu Translator

A professional English to Tulu translation system with both simple lookup and neural network models.

![Tulu Translator Interface](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Place your JSON files in the `dataset/` folder. Example format:
```json
[
  {"en": "Tiger", "tu": "Pili", "category": "animal"},
  {"en": "Father", "tu": "Ammer", "category": "family"}
]
```

### 3. Train the Model
```bash
python train.py
```
This creates:
- `model/tulu_translation_model.pth` (Neural network model)
- `model/tulu_lookup.pkl` (Lookup dictionary)

### 4. Start the Web Interface
```bash
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

## 📁 Project Structure
```
tulu-translator/
├── app.py              # Flask web server
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface
├── dataset/           # Your JSON translation files
└── model/            # Trained models (created after training)
```

## 🌐 Web Interface Features
- **Real-time Translation** - Translates as you type
- **Two Translation Modes**:
  - **Simple Lookup** - Exact matching (fast, 100% accurate)
  - **Neural Model** - AI-powered (handles variations)
- **Live Statistics** - Vocabulary size, translation count, categories
- **Translation History** - Keeps track of recent translations
- **Quick Examples** - One-click translation examples
- **Professional UI** - Modern, responsive design

## 🔧 API Endpoints
- `POST /translate` - Translate text
- `GET /stats` - Get system statistics
- `GET /categories` - Get dataset categories
- `GET /examples` - Get example phrases
- `GET /health` - Health check

## 🎯 Usage Examples

### 1. Using the Web Interface
1. Type English text in the input box
2. Choose translation mode (Simple/Neural)
3. Translation appears instantly
4. Check confidence score and history

### 2. Using API Directly
```bash
# Translate text
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Tiger", "method": "simple"}'

# Get statistics
curl http://localhost:5000/stats
```

## 📊 Supported Categories
- Animals (`animal.json`)
- Family (`family.json`)
- Greetings (`greeting.json`)
- Food (`food.json`)
- Transportation (`transportation.json`)
- ...and any other JSON files you add!

## 🛠️ Troubleshooting

### Issue: "Model not found"
```bash
# Make sure you ran train.py first
python train.py
```

### Issue: "No translations found"
- Check your JSON files are in `dataset/` folder
- Verify JSON format is correct
- Restart the app after adding new files

### Issue: "Port 5000 already in use"
```bash
# Change port in app.py
app.run(debug=True, port=5001)  # Change 5000 to 5001
```

## 📝 Adding New Translations

1. **Add to JSON file:**
```json
{
  "en": "Your English phrase",
  "tu": "Tulu translation",
  "category": "category_name"
}
```

2. **Retrain the model:**
```bash
python train.py
```

3. **Restart the app:**
```bash
python app.py
```

## 🎨 Customization

### Change UI Colors
Edit `templates/index.html`:
```css
/* Change primary color */
.bg-blue-500 { background-color: #your-color; }
.text-blue-500 { color: #your-color; }
```

### Add New Categories
Just add a new JSON file to `dataset/` folder:
```bash
dataset/
├── animals.json
├── family.json
├── your_new_category.json  # Add this
└── ...
```

## 🤝 Contributing
1. Fork the repository
2. Add your translations in JSON format
3. Submit a pull request