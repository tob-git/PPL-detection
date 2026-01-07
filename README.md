---
title: PPE Detection
emoji: 🦺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# PPE Detection - Complete Machine Learning Pipeline

This project implements a comprehensive Personal Protective Equipment (PPE) detection system using both deep learning (YOLO) and traditional machine learning approaches.

## Project Overview

The project meets all required machine learning pipeline components:

### ✅ I. Dataset with Train/Test Split
- **Dataset**: Construction PPE dataset with 11 classes
- **Training set**: 1,132 images
- **Validation set**: 143 images  
- **Test set**: 141 images
- **Classes**: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots

### ✅ II. Feature Extraction
Multiple feature types extracted from images:
- **Color Features**: RGB and HSV histograms (160 features)
- **Texture Features**: Sobel gradients, Laplacian variance, gray-level statistics
- **Shape Features**: Contour-based features, edge density
- **Statistical Features**: Mean, std, median, percentiles per channel

### ✅ III. Feature Selection
Three feature selection methods implemented:
- **SelectKBest**: ANOVA F-test based selection
- **PCA**: Principal Component Analysis for dimensionality reduction
- **RFE**: Recursive Feature Elimination with Random Forest

### ✅ IV. Multiple Classifiers
Six different classifiers implemented:
1. **Decision Tree**: Simple tree-based classifier
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting framework
4. **KNN**: K-Nearest Neighbors
5. **SVM**: Support Vector Machine with RBF kernel
6. **ANN (MLP)**: Multi-layer Perceptron neural network

### ✅ V. Performance Evaluation
Comprehensive metrics computed:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted precision
- **Recall**: Per-class and weighted recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Comparative Analysis**: Metrics across all classifier-feature combinations

## Project Structure

```
PPL detection/
├── ppe.py                      # YOLO-based detection pipeline
├── ml_pipeline.py              # Traditional ML classification pipeline ⭐
├── infer.py                    # YOLO inference script
├── inferonnx.py               # ONNX inference script
├── requirements.txt            # Python dependencies
├── construction-ppe.yaml       # Dataset configuration
├── datasets/
│   └── construction-ppe/
│       ├── images/            # Image files
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/            # YOLO format labels
│           ├── train/
│           ├── val/
│           └── test/
├── models/
│   ├── ppe_yolov8.pt          # Trained YOLO model
│   └── ppe_yolov8.onnx        # ONNX format model
└── outputs/
    ├── ml_pipeline/            # ML pipeline results
    │   ├── confusion_matrix_*.png
    │   ├── metrics_comparison.png
    │   ├── detailed_metrics.csv
    │   ├── features_train.pkl
    │   └── features_test.pkl
    ├── label_vis_train/        # Label visualizations
    └── label_vis_val/
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Traditional ML Pipeline (All Requirements) ⭐

Run the complete ML pipeline with feature extraction, selection, and multiple classifiers:

```bash
python ml_pipeline.py
```

This will:
1. ✅ Load and analyze the train/val/test dataset
2. ✅ Extract color, texture, shape, and statistical features
3. ✅ Apply multiple feature selection methods (SelectKBest, PCA, RFE)
4. ✅ Train 6 different classifiers (Decision Tree, RF, XGBoost, KNN, SVM, ANN)
5. ✅ Evaluate with accuracy, precision, recall, F1-score, and confusion matrices
6. ✅ Generate comprehensive visualizations and comparison plots

**Output**: All results saved to `outputs/ml_pipeline/`

### Option 2: YOLO Deep Learning Approach

Run the YOLO-based detection pipeline:

```bash
python ppe.py
```

This will:
- Analyze dataset statistics
- Generate label visualizations
- Train YOLOv8 model
- Validate and predict on test images

### Option 3: Run Inference Only

For quick inference with trained YOLO model:

```bash
python infer.py
```

## Results

After running `ml_pipeline.py`, you'll find:

### Outputs in `outputs/ml_pipeline/`:

1. **Confusion Matrices**: `confusion_matrix_<method>_<classifier>.png`
   - Visual representation of true vs predicted labels
   - One for each classifier-feature method combination
   
2. **Metrics Comparison**: `metrics_comparison.png`
   - Bar plots comparing Accuracy, Precision, Recall, F1-Score
   - Across all classifiers and feature selection methods
   
3. **Detailed Metrics**: `detailed_metrics.csv`
   - Complete numerical results
   - Easy to import into Excel/Google Sheets for further analysis
   
4. **Cached Features**: `features_train.pkl`, `features_test.pkl`
   - Extracted features saved for quick re-runs
   - Delete these to force re-extraction

### Sample Results Format:

```
SUMMARY: Best Performing Models
================================================
Best Accuracy:
  Classifier: Random Forest
  Feature Method: pca
  Accuracy: 0.8523

Best F1-Score:
  Classifier: XGBoost
  Feature Method: kbest
  F1-Score: 0.8467
```

## Key Features

### Comprehensive Feature Extraction
- **160 color histogram features** (BGR + HSV)
- **9 texture features** (Sobel, Laplacian, gray statistics)
- **8 shape features** (contours, edges)
- **15 statistical features** (per-channel statistics)
- **Total: ~192 features per image**

### Robust Feature Selection
- Reduces dimensionality while maintaining performance
- Compares 3 different selection strategies
- Prevents overfitting on high-dimensional data

### Multiple Classifier Comparison
- **Tree-based**: Decision Tree, Random Forest, XGBoost
- **Instance-based**: KNN
- **Kernel-based**: SVM
- **Neural Network**: Multi-layer Perceptron
- Find the best model for your specific use case

### Comprehensive Evaluation
- Multiple metrics beyond just accuracy
- Confusion matrices for error analysis
- Visual comparisons across all combinations
- Exportable results for reports/presentations

## Verification Checklist

- [x] **I. Dataset Split**: ✅ Train (1132) / Val (143) / Test (141)
- [x] **II. Feature Extraction**: ✅ Color, Texture, Shape, Statistical features
- [x] **III. Feature Selection**: ✅ SelectKBest, PCA, RFE
- [x] **IV. Multiple Classifiers**: ✅ Decision Tree, RF, XGBoost, KNN, SVM, ANN
- [x] **V. Evaluation Metrics**: ✅ Accuracy, Precision, Recall, F1, Confusion Matrix

## Troubleshooting

### Out of Memory
If you encounter memory issues:
- Reduce image size in `extract_all_features()` (default: 128x128)
- Process fewer images for testing
- Use feature selection methods that reduce dimensionality more

### Slow Training
- SVM and ANN can be slow on large datasets
- Start with Random Forest or XGBoost for faster results
- Use cached features (`.pkl` files) for subsequent runs

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## Extensions and Improvements

Potential enhancements:
1. **Deep Features**: Extract features from pre-trained CNNs (ResNet, VGG)
2. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
3. **Ensemble Methods**: Combine multiple classifiers with voting
4. **Class Imbalance**: Handle with SMOTE or class weights
5. **Cross-Validation**: K-fold CV for more robust evaluation

## License

This project uses the Ultralytics YOLO model which is licensed under AGPL-3.0.
Dataset: Construction-PPE by Ultralytics

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Author**: Mohamd Tobgi  
**Course**: Introduction to Machine Learning - Fall 2026  
**Project**: PPE Detection with Traditional ML Pipeline
