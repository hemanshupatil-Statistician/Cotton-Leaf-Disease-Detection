# 🧬 Cotton Leaf Disease Detection

This project applies **deep learning**, specifically a **Convolutional Neural Network (CNN)**, to detect common diseases in cotton leaves. Built and trained in **Google Colab**, the model can classify images into four categories based on leaf condition.

---

## 📂 Dataset

The dataset was sourced from Kaggle:

- 🔸 [Cotton Leaf Disease Dataset by SeroshKarim](https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset)

It includes four classes:
- **Bacterial Blight**
- **Curl Virus**
- **Fusarium Wilt**
- **Healthy**

---

## 🧠 Model Details

- **Model type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input size:** 150x150 color images
- **Output:** Softmax prediction over 4 classes

The model was trained with image augmentation techniques including:
- Rescaling
- Horizontal flipping
- Zoom
- Shear transformations

---

## 🚀 How to Use

### ▶️ Run the Notebook

- Clone or download the repo
- Open the notebook (`cotton.ipynb`) in [Google Colab](https://colab.research.google.com)
- Download the trained model and dataset using the links below

### 🔗 Model and Dataset Downloads

Because of GitHub’s file size restrictions, the model and dataset are hosted externally:

- 🔸 [Download Trained Model (`cotton.keras`)]([https://drive.google.com/file/d/YOUR-MODEL-ID/view?usp=sharing](https://drive.google.com/file/d/15ReHdRr2hkYR7YRILPXr8oWB0ZRGJZmU/view?usp=drive_link))
- 🔸 [Download Dataset (`cotton-leaf-disease-dataset`)]([https://drive.google.com/file/d/YOUR-DATASET-ID/view?usp=sharing](https://drive.google.com/drive/folders/1EAkQClIh005AqnCQt6weAAQSRjIlX7w8?usp=drive_link))

> 💡 Replace the Google Drive links above with actual shared links (ensure access is set to "Anyone with the link").

### 🧪 Sample Inference Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cotton.keras')  # Ensure it's in your working directory

# Load and preprocess test image
img = image.load_img('test_leaf.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
class_labels = ['Bacterial Blight', 'Curl Virus', 'Fusarium Wilt', 'Healthy']
prediction = model.predict(img_array)
print("Predicted Class:", class_labels[np.argmax(prediction)])

