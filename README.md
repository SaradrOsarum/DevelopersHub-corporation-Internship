# Disease Detection Models

This repository contains the work completed during my three-week internship, focusing on developing machine learning models to detect diseases from various types of data. Each week was dedicated to a specific project, demonstrating skills in data preprocessing, model training, evaluation, and deployment.

## Weekly Projects

### Week 1: Disease Prediction Using Patient Data
- **Objective**: Predict diseases based on structured patient data.
- **Steps**:
  - Preprocessed data, handled imbalanced datasets using `SMOTE`.
  - Applied feature scaling and engineering techniques.
  - Trained machine learning models like Random Forest and Gradient Boosting.
  - Evaluated model performance and tuned hyperparameters.
- **Libraries**: scikit-learn, Imbalanced-learn, Pandas, Seaborn, Matplotlib.

### Week 2: Cancer Detection Using Histopathological Images
- **Objective**: Detect cancer in histopathological images.
- **Steps**:
  - Preprocessed image data and prepared for model training.
  - Used transfer learning with `VGG16` for feature extraction.
  - Highlighted cancerous regions in the images.
  - Evaluated model performance.
- **Libraries**: TensorFlow, scikit-learn, Matplotlib, KaggleHub.

### Week 3: Skin Cancer Detection and Pneumonia Detection from Chest X-Rays
#### Skin Cancer Detection
- **Objective**: Classify skin images as cancerous or non-cancerous.
- **Steps**:
  - Loaded and preprocessed the dataset.
  - Applied data augmentation to improve generalization.
  - Defined and trained a deep learning model using `ResNet50`.
  - Fine-tuned and evaluated the model's performance.
- **Libraries**: TensorFlow, scikit-learn, Matplotlib, Pandas.

#### Pneumonia Detection from Chest X-Rays
- **Objective**: Detect pneumonia from chest X-ray images.
- **Steps**:
  - Downloaded and preprocessed data using Kaggle's API.
  - Used `ImageDataGenerator` for augmentation.
  - Built and trained a CNN model with `MobileNetV2`.
  - Visualized model performance and saved results.
- **Libraries**: TensorFlow, scikit-learn, JSON, Matplotlib.

## Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab
- Libraries:
  - TensorFlow
  - scikit-learn
  - Pandas
  - Matplotlib
  - Seaborn
  - Imbalanced-learn
  - Kaggle API

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disease-detection-models.git
   cd disease-detection-models
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the desired Jupyter Notebook.
2. Run all cells to preprocess data, train the model, and evaluate its performance.
3. Modify paths or parameters if needed to customize for your dataset.

## Results
- Each notebook includes visualizations and metrics to assess the model's performance.
- Saved models and outputs are available for further deployment or analysis.

## Contributing
Contributions are welcome! Feel free to fork this repository, make improvements, and submit a pull request.


---

### Acknowledgments
- Data sources: Kaggle datasets for medical imaging and structured data.
- Inspiration: Application of deep learning and machine learning for healthcare solutions.

