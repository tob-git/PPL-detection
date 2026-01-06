"""
PPE Detection App - YOLO Detection + ML Classification
Combines YOLO object detection with traditional ML classifiers
"""

import streamlit as st
import cv2
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Configuration
OUTPUT_DIR = Path("outputs/ml_pipeline")
YOLO_MODEL_PATH = Path("models/ppe_yolov8.pt")

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    'helmet': (0, 255, 0), 'gloves': (0, 255, 0), 'vest': (0, 255, 0),
    'boots': (0, 255, 0), 'goggles': (0, 255, 0),
    'no_helmet': (0, 0, 255), 'no_goggle': (0, 0, 255),
    'no_gloves': (0, 0, 255), 'no_boots': (0, 0, 255),
    'Person': (255, 255, 0), 'none': (128, 128, 128)
}

@st.cache_resource(ttl=60)  # Refresh every 60 seconds during debugging
def load_models():
    """Load all models and feature selectors."""
    models = {}
    
    # Feature methods available
    feature_methods = ['original', 'kbest', 'pca', 'rfe']
    
    # Classifier names
    classifier_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', 'ANN (MLP)']
    
    # Load all model combinations
    for clf_name in classifier_names:
        safe_clf = clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        for feat_method in feature_methods:
            model_key = f"{clf_name} ({feat_method})"
            model_path = OUTPUT_DIR / f"{safe_clf}_model_{feat_method}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[model_key] = pickle.load(f)
                    print(f"Loaded model: {model_key}")
            else:
                print(f"Model not found: {model_path}")
    
    # Add YOLO as an option
    models['YOLO'] = 'YOLO'
    
    # Load feature selectors
    selectors = {'original': None}  # Original has no selector
    for feat_method in ['kbest', 'pca', 'rfe']:
        selector_path = OUTPUT_DIR / f"selector_{feat_method}.pkl"
        if selector_path.exists():
            with open(selector_path, 'rb') as f:
                selectors[feat_method] = pickle.load(f)
                print(f"Loaded selector: {feat_method}")
        else:
            print(f"Warning: Selector not found: {selector_path}")
            selectors[feat_method] = None
    
    # Load scaler and label encoder
    with open(OUTPUT_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(OUTPUT_DIR / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    yolo_model = YOLO(str(YOLO_MODEL_PATH)) if YOLO_MODEL_PATH.exists() else None
    
    return models, selectors, scaler, label_encoder, yolo_model

def extract_hog_features(img, target_size=(64, 64)):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    win_size = target_size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    
    return features.flatten()

def extract_lbp_features(img, target_size=(64, 64), radius=1, n_points=8):
    """Extract LBP (Local Binary Pattern) features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    lbp = np.zeros_like(gray)
    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            binary = 0
            for n in range(n_points):
                angle = 2 * np.pi * n / n_points
                x = int(round(j + radius * np.cos(angle)))
                y = int(round(i - radius * np.sin(angle)))
                if gray[y, x] >= center:
                    binary |= (1 << n)
            lbp[i, j] = binary
    
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_color_features(img, target_size=(64, 64)):
    """Extract enhanced color histogram features."""
    img = cv2.resize(img, target_size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    for i, (bins, range_max) in enumerate([(32, 180), (32, 256), (32, 256)]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, range_max])
        features.extend(hist.flatten())
    
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    for channel in cv2.split(img):
        features.extend([np.mean(channel), np.std(channel), 
                        np.mean((channel - np.mean(channel))**3)])
    
    features = np.array(features)
    features = features / (np.linalg.norm(features) + 1e-7)
    
    return features

def extract_texture_features(img, target_size=(64, 64)):
    """Extract enhanced texture features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    features.extend([
        np.mean(magnitude), np.std(magnitude), np.max(magnitude),
        np.mean(direction), np.std(direction)
    ])
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.var(laplacian))
    
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 3]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
    
    features.extend([
        np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
        np.percentile(gray, 25), np.percentile(gray, 75)
    ])
    
    return np.array(features)

def extract_shape_features(img, target_size=(64, 64)):
    """Extract shape-based features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = [len(contours)]
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)
        else:
            features.extend([0] * 7)
        
        features.extend([
            np.mean(areas), np.std(areas), np.max(areas), np.sum(areas),
            np.mean(perimeters), np.std(perimeters), np.max(perimeters)
        ])
    else:
        features.extend([0] * 14)
    
    features.append(np.sum(edges > 0) / edges.size)
    
    return np.array(features)

def extract_all_features(roi, target_size=(64, 64)):
    """Extract ALL enhanced features from ROI."""
    roi = cv2.resize(roi, target_size)
    
    hog = extract_hog_features(roi, target_size)
    lbp = extract_lbp_features(roi, target_size)
    color = extract_color_features(roi, target_size)
    texture = extract_texture_features(roi, target_size)
    shape = extract_shape_features(roi, target_size)
    
    features = np.concatenate([hog, lbp, color, texture, shape])
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    
    return features

def detect_and_classify(image, classifier, feature_selector, scaler, label_encoder, yolo_model, conf_threshold=0.25, use_yolo_class=False):
    """Detect with YOLO and classify with ML model or YOLO."""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_display = img.copy()
    detections = []
    
    # YOLO class names mapping
    yolo_class_names = {
        0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
        5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
        9: "no_gloves", 10: "no_boots"
    }
    
    if yolo_model:
        results = yolo_model(img, conf=conf_threshold, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = img[y1:y2, x1:x2]
                
                if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                    # Determine class and confidence
                    if use_yolo_class:
                        # Use YOLO's prediction
                        yolo_cls = int(box.cls[0])
                        pred_class = yolo_class_names.get(yolo_cls, f"class_{yolo_cls}")
                        pred_conf = float(box.conf[0])
                    else:
                        # Use ML classifier with feature selection
                        features = extract_all_features(roi)
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # Apply feature selection if provided
                        if feature_selector is not None:
                            features_scaled = feature_selector.transform(features_scaled)
                        
                        # Debug: print feature dimensions
                        print(f"Features shape after selection: {features_scaled.shape}")
                        
                        pred_encoded = classifier.predict(features_scaled)[0]
                        pred_class = label_encoder.inverse_transform([pred_encoded])[0]
                        
                        if hasattr(classifier, 'predict_proba'):
                            proba = classifier.predict_proba(features_scaled)[0]
                            pred_conf = float(proba[pred_encoded])
                        else:
                            pred_conf = float(box.conf[0])
                    
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'class': pred_class,
                        'confidence': pred_conf
                    })
                    
                    # Draw
                    color = CLASS_COLORS.get(pred_class, (255, 255, 255))
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{pred_class}: {pred_conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_display, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    cv2.putText(img_display, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    return img_display, detections

def main():
    st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")
    
    st.title("🦺 PPE Detection: YOLO + ML Classifiers")
    st.markdown("""
    **How it works:**
    1. YOLO detects objects and bounding boxes
    2. Your chosen ML classifier identifies each object
    3. Results visualized with colored boxes
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            models, selectors, scaler, label_encoder, yolo_model = load_models()
            if not models or not yolo_model:
                st.error("Models not found! Run: `python3 ml_pipeline.py`")
                return
            st.success(f"✅ Loaded {len(models)} model combinations")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Feature selection method dropdown
    feature_method = st.sidebar.selectbox(
        "Feature Selection Method",
        options=['original', 'kbest', 'pca', 'rfe'],
        format_func=lambda x: {
            'original': 'Original (All Features - 2361)',
            'kbest': 'SelectKBest (Top 100)',
            'pca': 'PCA (100 Components)',
            'rfe': 'RFE (80 Features)'
        }[x]
    )
    
    # Classifier dropdown
    classifier_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', 'ANN (MLP)', 'YOLO']
    selected_classifier = st.sidebar.selectbox("Classifier", classifier_names)
    
    # Construct full model key
    if selected_classifier == 'YOLO':
        selected_model = 'YOLO'
        use_yolo = True
    else:
        selected_model = f"{selected_classifier} ({feature_method})"
        use_yolo = False
    
    conf_threshold = 0.25  # Fixed confidence threshold
    
    # Upload
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original")
            st.image(image, use_container_width=True)
        
        with st.spinner(f"Processing with {selected_classifier} ({feature_method if not use_yolo else 'N/A'})..."):
            # Get feature selector based on selected method
            feature_selector = selectors.get(feature_method, None) if not use_yolo else None
            
            # Debug info
            if not use_yolo:
                st.sidebar.write(f"🔍 Using: {selected_classifier} with {feature_method} features")
                st.sidebar.write(f"🔍 Selector: {type(feature_selector).__name__ if feature_selector else 'None (original features)'}")
            
            # Check if model exists
            if selected_model not in models:
                st.error(f"Model '{selected_model}' not found! Run ml_pipeline.py first.")
                return
            
            result_img, detections = detect_and_classify(
                image, 
                models[selected_model] if not use_yolo else None,
                feature_selector,
                scaler, 
                label_encoder, 
                yolo_model, 
                conf_threshold,
                use_yolo_class=use_yolo
            )
            
            with col2:
                st.subheader(f"🔍 Results ({selected_model})")
                st.image(result_img, use_container_width=True)
        
        # Stats
        st.markdown("---")
        if detections:
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Total Detections", len(detections))
            
            class_counts = {}
            for d in detections:
                class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
            
            safe = sum(class_counts.get(c, 0) for c in ['helmet', 'gloves', 'vest', 'boots', 'goggles'])
            unsafe = sum(class_counts.get(c, 0) for c in ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots'])
            
            with cols[1]:
                st.metric("✅ Safe PPE", safe)
            with cols[2]:
                st.metric("⚠️ Missing PPE", unsafe, delta=f"-{unsafe}" if unsafe > 0 else None, delta_color="inverse")
            with cols[3]:
                safety_score = (safe / (safe + unsafe) * 100) if (safe + unsafe) > 0 else 0
                st.metric("Safety Score", f"{safety_score:.0f}%")
            
            # Table
            st.subheader("📊 Detected Objects")
            import pandas as pd
            df = pd.DataFrame(detections)
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            df = df[['class', 'confidence']]
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No objects detected")
    else:
        st.info("👆 Upload an image to start!")

if __name__ == "__main__":
    main()
