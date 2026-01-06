# 🦺 PPE Detection UI - Quick Start Guide

## Overview

The PPE Detection UI allows you to:
- ✅ Upload images
- ✅ Choose between YOLO or Traditional ML models
- ✅ See real-time predictions with visualizations
- ✅ Compare different classifiers

## Setup Instructions

### Step 1: Install Streamlit

```bash
pip3 install streamlit
```

Or install all dependencies:

```bash
pip3 install -r requirements.txt
```

### Step 2: Train and Save Models

If you haven't already, run the improved ML pipeline to train and save models:

```bash
python3 ml_pipeline.py
```

This will:
- Train 6 different classifiers
- Save all models to `outputs/ml_pipeline/`
- Save the scaler and label encoder

### Step 3: Launch the UI

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Using the UI

### 1. Upload an Image
- Click "Browse files" or drag and drop an image
- Supported formats: JPG, JPEG, PNG
- Best results with clear images showing workers and PPE

### 2. Choose Model Type

**Option A: YOLO (Object Detection)**
- Detects and localizes multiple PPE items
- Shows bounding boxes around detected objects
- Provides confidence scores for each detection
- Best for images with multiple workers/objects

**Option B: Traditional ML (Classification)**
- Classifies entire image as "Safe PPE" or "Unsafe"
- Choose from 6 different classifiers:
  - Random Forest
  - XGBoost
  - SVM
  - Decision Tree
  - KNN
  - ANN (MLP)
- Good for overall safety assessment

### 3. View Results
- See predictions with confidence scores
- Visual feedback (✅ for safe, ⚠️ for unsafe)
- Annotated images showing detections

## Features

### 🎨 Visual Interface
- Clean, modern design
- Side-by-side image comparison
- Color-coded results

### 📊 Model Comparison
- Test different classifiers on the same image
- Compare accuracy and confidence
- See which model works best

### 🔧 Model Status
- Check which models are loaded
- See if YOLO model is available
- Count of available ML models

## Troubleshooting

### "YOLO model not found"
**Solution:** Train the YOLO model first:
```bash
python3 ppe.py
```

### "ML models not found"
**Solution:** Run the improved ML pipeline:
```bash
python3 ml_pipeline.py
```

### "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install Streamlit:
```bash
pip3 install streamlit
```

### Port already in use
**Solution:** Use a different port:
```bash
streamlit run app.py --server.port 8502
```

## Tips for Best Results

### Image Quality
- Use well-lit images
- Clear view of workers and PPE
- Avoid blurry or low-resolution images

### YOLO Model
- Best for detecting specific PPE items
- Works with multiple objects
- Shows locations of each item

### ML Models
- Best for overall safety assessment
- Faster inference than YOLO
- Good for binary decisions (safe/unsafe)

## Model Performance

Based on our testing:

| Model | Accuracy | Best For |
|-------|----------|----------|
| YOLO | High | Object detection, multiple items |
| Random Forest | ~70-80% | Binary classification |
| XGBoost | ~70-80% | Binary classification |
| SVM | ~70-80% | Binary classification |
| KNN | ~60-70% | Simple cases |
| Decision Tree | ~60-70% | Interpretability |
| ANN (MLP) | ~60-75% | Complex patterns |

## Example Workflow

1. **Upload** a construction site image
2. **Try YOLO** to see all detected PPE items
3. **Switch to ML** models to get safety classification
4. **Compare** different classifiers
5. **Choose** the model that works best for your use case

## Advanced Options

### Custom Model Paths

Edit `app.py` to change model locations:

```python
MODELS_DIR = Path("your/custom/path")
OUTPUT_DIR = Path("your/output/path")
```

### Add New Models

1. Train your model
2. Save as pickle file in `outputs/ml_pipeline/`
3. Add to model list in `app.py`

### Change Confidence Threshold

For YOLO predictions, edit in `app.py`:

```python
results = model(img_array, conf=0.25)  # Change 0.25 to your threshold
```

## Keyboard Shortcuts

- **Ctrl+R** / **Cmd+R**: Reload app
- **Ctrl+C** in terminal: Stop server

## Deployment

### Local Network Access

Allow others on your network to access:

```bash
streamlit run app.py --server.address 0.0.0.0
```

### Share Externally

Use Streamlit Cloud for free hosting:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Deploy your app

## Support

For issues or questions:
- Check error messages in terminal
- Review model training logs
- Ensure all files are in correct directories

## File Structure

```
PPL detection/
├── app.py                    # Main UI application ⭐
├── ml_pipeline.py   # Training script
├── ppe.py                    # YOLO training
├── outputs/
│   └── ml_pipeline/
│       ├── *_model.pkl       # Saved classifiers
│       ├── scaler.pkl        # Feature scaler
│       └── label_encoder.pkl # Label encoder
└── models/
    └── ppe_yolov8.pt         # YOLO model
```

## Next Steps

- Try different test images
- Compare model performances
- Fine-tune confidence thresholds
- Add more classifiers
- Customize the UI

---

**Ready to test?** Run: `streamlit run app.py` 🚀
