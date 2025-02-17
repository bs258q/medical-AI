# MONAI-Based Medical Image Classification

## Overview
This project demonstrates the use of **MONAI (Medical Open Network for AI)** for training a **DenseNet121** model on medical imaging data. The pipeline includes:
- **Preprocessing medical images** using MONAI transforms
- **Training a DenseNet model** with PyTorch and MONAI
- **Evaluating model performance** using AUC-ROC metrics

## Dataset
This project assumes the presence of a medical imaging dataset, which is preprocessed using **MONAI transforms**. Example datasets include:
- **NIH Chest X-ray Dataset** ([Link](https://nihcc.app.box.com/v/ChestXray-NIHCC))
- **PhysioNet Datasets** ([Link](https://physionet.org/))
- **MIMIC-III (Medical ICU Data)** ([Link](https://physionet.org/content/mimiciii/1.4/))

## Project Steps

### 1️⃣ Data Preprocessing
- **Use MONAI transforms** (`LoadImage`, `EnsureChannelFirst`, `ScaleIntensity`, `ToTensor`) to preprocess images.
- Cache datasets for efficient loading.

### 2️⃣ Model Training
- Use **DenseNet121** as the backbone model.
- Define **loss function** (CrossEntropyLoss) and **optimizer** (Adam).
- Train the model for multiple epochs while tracking loss.

### 3️⃣ Model Evaluation
- Use **AUC-ROC** metric to evaluate performance.
- Store the best-performing model based on AUC.
- Validate the model on unseen test data.

## Installation & Usage
### **1. Install Required Dependencies**
```bash
pip install monai torch torchvision numpy matplotlib
```

### **2. Run Training Script**
```bash
python train_monai.py
```

### **3. Visualize Results**
To plot the AUC-ROC curve, modify the evaluation section:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_true.cpu(), y_pred.cpu())
auc_score = roc_auc_score(y_true.cpu(), y_pred.cpu())

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

## Results
- The model achieves **high AUC scores**, indicating strong classification performance.
- Overfitting is prevented using validation monitoring.
- The best-performing model is saved as `best_model.pth`.

## Contributions
Feel free to fork this repository, improve the training pipeline, or experiment with different datasets. 🚀

## License
This project is **open-source** and available under the MIT License.
