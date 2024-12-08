# Corneal Ulcer Detection Using U-Net and ResNet50

This project implements a Corneal Ulcer Detection model using a U-Net architecture. The model is trained to segment corneal ulcers from raw images using deep learning techniques. The primary focus is on providing accurate segmentation to aid medical diagnosis.

---

## Features

- **U-Net Architecture:** A state-of-the-art convolutional neural network architecture tailored for image segmentation tasks.
- **Binary Segmentation:** Outputs binary masks to identify ulcer regions in corneal images.
- **Custom Training Pipeline:** Implements custom data preprocessing and training from scratch.

---

## Requirements

Ensure you have the following dependencies installed:

- Python 3.7+
- TensorFlow
- OpenCV
- Matplotlib
- NumPy
- scikit-learn

Install the required libraries using:

```bash
pip install tensorflow opencv-python matplotlib numpy scikit-learn
```

---

## Dataset

The dataset consists of:

- **Raw Images:** High-resolution corneal images.
- **Ulcer Labels:** Corresponding binary masks (grayscale images) indicating ulcer regions.

Place the datasets in the following structure:

```
/kaggle/input/raw-images
/kaggle/input/ulcerlabels
```

---

## Usage

### Step 1: Load the Dataset

The `load_data` function reads and preprocesses images and labels:

- Normalizes pixel values to [0, 1].
- Resizes images and masks to `(224, 224)`.

### Step 2: Train the Model

Run the training script:

```python
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=16,
    epochs=50
)
```

### Step 3: Evaluate the Model

Evaluate the trained model on test data:

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Step 4: Visualize Predictions

Predict and visualize segmentation masks for a given image:

```python
predict_and_visualize(image, model)
```

---

## Results

- **Accuracy:** Achieved test accuracy (displayed after training).
- **Loss:** Optimized using Binary Crossentropy.
- **Visualization:** The segmentation masks can be visualized alongside the original images.

---

## Hyperparameters

- **Input Image Size:** `224x224x3`
- **Batch Size:** `16`
- **Learning Rate:** `0.0001`
- **Epochs:** `50`

---

## Visualization Example

```python
predict_and_visualize(image, model)
```

- **Left:** Original Image.
- **Right:** Predicted Segmentation Mask.

---

## Future Enhancements

1. **Data Augmentation:** Improve model generalization.
2. **Fine-tuning Encoder:** Enable ResNet50 layers for fine-tuning.
3. **Advanced Architectures:** Experiment with ResNet18, EfficientNet, or custom backbones.

---

## License

This project is open-source and distributed under the MIT License.

---

## Contributors

If you encounter any issues or have suggestions, feel free to contribute or raise an issue.
