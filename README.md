Real vs AI-Generated Image Detection
->Project Overview

This project detects whether a given image is Real or AI-Generated using Deep Learning.
With the rapid growth of AI-generated content and deepfakes, verifying image authenticity has become essential.
The system uses Transfer Learning with MobileNetV2 to classify images as real or synthetic.

->Model Architectur
Pretrained Model: MobileNetV2 (ImageNet weights)
Input Size: 224 × 224 × 3
Output: Binary Classification (Real / Fake)
Activation Function: Sigmoid

Optimizer: Adam
Loss Function: Binary Crossentropy
Technique: Transfer Learning + Data Augmentation

-> Dataset
Dataset: RealVsFake-81K (Kaggle)
Classes: Real Images, AI-Generated Images
Split: 80% Training / 20% Validation

->Features

✔ AI-Generated Image Detection
✔ Transfer Learning Implementation
✔ Data Augmentation for Better Generalization
✔ Binary Classification System
✔ Streamlit Deployment Ready

🛠️ Technologies Used
Python • TensorFlow/Keras • MobileNetV2 • NumPy • Matplotlib • Streamlit • Google Colab
=> Training Details
Batch Size: 32
Image Size: 224 × 224
Epochs: 5
Augmentation: Rotation, Horizontal Flip, Zoom

->Workflow
Upload Image
Resize & Normalize
Model Predicts Probability
Output: Real or Fake
