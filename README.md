# 🌿 Smart Crop Health Monitoring System

An AI-powered plant disease detection system that analyzes crop leaf
images and recommends treatment — built using Transfer Learning
and a comparative study of MobileNetV2 vs ResNet50.

## What it does

- Upload a crop leaf image
- Detects plant disease using a fine-tuned MobileNetV2 model
- Returns disease name, confidence score, and recommended treatment
- Supports Tomato Early Blight, Potato Early Blight, Late Blight,
  and Healthy classifications


## Tech Stack

- **Models**: MobileNetV2, ResNet50 (Transfer Learning, ImageNet weights)
- **Dataset**: PlantVillage Dataset (Kaggle)
- **Framework**: TensorFlow / Keras
- **Frontend**: Streamlit
- **Libraries**: NumPy, Matplotlib, Pillow

## Model Comparison

| Model       | Training Accuracy | Validation Accuracy | Training Time |
|-------------|-------------------|---------------------|---------------|
| MobileNetV2 | 95.89%            | 91.24%              | 212s          |
| ResNet50    | 49.46%            | 51.80%              | 328s          |

## Key Findings

MobileNetV2 significantly outperformed ResNet50 on this dataset —
achieving 91.24% validation accuracy in 212s versus ResNet50's
51.80% in 328s. ResNet50's poor performance is likely due to its
deeper architecture requiring more epochs and data augmentation to
converge effectively. MobileNetV2 was selected for the final
Streamlit deployment given its superior accuracy and faster
inference speed — making it more suitable for real-time crop
monitoring in low-resource environments.

## Model Architecture

Both models use Transfer Learning with frozen ImageNet weights,
with a custom classification head fine-tuned on PlantVillage.

**MobileNetV2** — lightweight, fast, selected for deployment
- GlobalAveragePooling2D → Dense(128, ReLU) → Dense(num_classes, Softmax)

**ResNet50** — deeper architecture, used for comparison
- GlobalAveragePooling2D → Dense(128, ReLU) → Dense(num_classes, Softmax)

Optimizer: Adam | Loss: Categorical Crossentropy | Epochs: 5

## Setup & Usage

### 1. Clone the repo
git clone https://github.com/yourusername/crop-health-monitoring
cd crop-health-monitoring
pip install -r requirements.txt

### 2. Download the dataset via Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d dataset/

### 3. Train the model
Run the notebook end to end in Google Colab or Jupyter.
Save the trained model: model.save('model.h5')

### 4. Run the Streamlit app
streamlit run app.py

If running on Colab, expose via localtunnel:
npx localtunnel --port 8501

## Dataset

PlantVillage Dataset — 54,000+ labeled leaf images across
38 plant disease categories, accessed via Kaggle API.

## Future Improvements

- Fine-tune ResNet50 with data augmentation and more epochs
- Expand to more crop and disease categories
- Deploy on Hugging Face Spaces for public access
- Add real-time camera input support
