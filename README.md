# Deep Learning Projects – From Scratch & TensorFlow/Keras Implementations

Welcome to my Deep Learning repository! 🚀  
This repo showcases a series of deep learning projects where I built models from scratch and using TensorFlow/Keras, with strong focus on model documentation, data analysis, and insightful visualizations.

## ✨ Highlights

- **Artificial Neural Networks (ANN)**
  - ANN built **from scratch** (no deep learning libraries).
  - ANN using Keras on the **Credit Score** dataset.

- **Convolutional Neural Networks (CNN)**
  - Custom CNN architectures on **CIFAR-10** and **MNIST** datasets.
  - Implementation of popular CNN models: **ResNet**, **VGG**, **U-Net**.

- **Image Segmentation**
  - **Chest CT scans** segmentation.
  - **Oxford-IIIT Pet Dataset** segmentation.

- **Recurrent Neural Networks (RNN)**
  - **LSTM** and **ConvLSTM** architectures applied on the **MNIST** dataset.

- **Data Insights & Visualization**
  - Extensive exploration of datasets.
  - Visualizations for predictions, learning curves, and evaluation metrics.

## 📚 Datasets Used

| Dataset         | Task                  | Description                                |
|-----------------|------------------------|--------------------------------------------|
| CIFAR-10        | Image Classification    | 10 classes of natural images.             |
| MNIST           | Image Classification    | Handwritten digit images.                 |
| Chest CT Scans  | Image Segmentation      | Medical imaging for lung segmentation.    |
| Oxford Pets     | Image Segmentation      | Segmentation of pet breeds with masks.    |
| Credit Score    | Tabular Classification  | Customer credit scoring prediction.       |

## 🛠 Technologies Used

- Python 3
- TensorFlow 2.x / Keras
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Pandas

## 📊 Sample Visualizations
- **Here are examples of visualizations generated during training and evaluation:**

  - Training vs Validation Curves
  
  - Plotting loss and accuracy curves over epochs to monitor overfitting and underfitting.
  
  - Visual interpretation of model classification results (especially for CIFAR-10 and MNIST).
  
  - Segmentation Masks
  
  - Overlaying predicted masks on original images for Chest CT scans and Oxford Pets datasets.
  
  - Predicted vs True Labels
  
  - Displaying side-by-side comparisons between actual and predicted outcomes.

- **Example Plots:**

  - Accuracy vs Epochs
  
  - Loss vs Epochs
  
  - Original Image | Ground Truth Mask | Predicted Mask
  
  - Sample LSTM predictions on sequences

(You can find these inside each notebook after training sections.)

## 🧠 Notes
- All models and experiments are documented step-by-step inside the notebooks.

- Training processes include evaluation metrics, and loss/accuracy visualization.

- Transfer Learning techniques are explored where relevant (especially with CNN models).

- Best practices like data augmentation, normalization, and batch optimization have been applied.

- Clear file organization: Each model/task is isolated in its own notebook for clarity.

- Code and documentation aim to be beginner-friendly and educational for new learners too
