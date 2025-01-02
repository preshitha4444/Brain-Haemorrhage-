
# Deep Learning Based Neural Network Algorithm for Intracranial Hemorrhage Detection

This project implements a deep learning-based neural network algorithm for detecting intracranial hemorrhage (ICH) from CT scan images. The model uses the YOLO (You Only Look Once) architecture, trained to classify CT scans as either "normal" or "hemorrhage." The system is built using Python and utilizes popular libraries like Keras, OpenCV, Tkinter, Matplotlib, and Scikit-learn.

## Features

- **Dataset Upload**: Upload a dataset of CT scan images for training and testing.
- **Image Preprocessing**: Normalizes the CT scan images, splits the dataset into training and testing sets.
- **YOLO Model Training**: Trains a deep learning YOLO model on the CT scan dataset.
- **Model Evaluation**: Evaluates the model's performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
- **Prediction**: Predicts the presence of hemorrhage in a new CT scan image.
- **Visualization**: Displays accuracy/loss graphs and confusion matrix heatmaps for model evaluation.
- **GUI Interface**: A user-friendly Tkinter-based interface to interact with the model.

## Requirements

To run this project, make sure to install the following dependencies:

- Python 3.7
- Keras
- TensorFlow
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn
- Tkinter (usually pre-installed with Python)

You can install the necessary libraries using pip:

```bash
pip install keras tensorflow opencv-python scikit-learn matplotlib seaborn
```

## File Structure

```
├── yolomodel/
│   ├── yolomodel.json            # YOLO model architecture
│   ├── yolomodel_weights.h5      # Pre-trained YOLO model weights
│   ├── X.txt.npy                # Processed CT scan images (features)
│   └── Y.txt.npy                # Corresponding labels (normal/hemorrhage)
├── main.py                       # Main script for training and predicting
├── history.pckl                  # Training history (accuracy, loss)
└── README.md                     # Project description and documentation
```

## Usage

1. **Upload the CT scan dataset**:
   - Use the "Upload CT Scans Dataset" button in the GUI to load your dataset directory.
   
2. **Preprocess the images**:
   - Click on "Normalize CT Scans Images" to preprocess the data (normalization and splitting into training/testing sets).

3. **Train the YOLO model**:
   - Click on "Train Yolo Model" to train the model on the preprocessed dataset. If a pre-trained model is available, it will be loaded, and training will resume.

4. **Evaluate the model**:
   - After training, the "Yolo Accuracy-Loss Graph" button will display the accuracy and loss curves.
   - The confusion matrix will be shown for model evaluation.

5. **Make predictions**:
   - Click on "Predict Hemorrhage from Test Image" to upload a CT scan image and classify it as either "Normal" or "Hemorrhage."

6. **Exit**:
   - Click on "Exit" to close the application.

## Model Details

The model is a Convolutional Neural Network (CNN) built using the Keras library. It consists of multiple convolutional and pooling layers followed by fully connected layers to classify images into one of two classes:

- Normal
- Hemorrhage

The model is compiled with the Adam optimizer and categorical crossentropy loss function. It is trained for 10 epochs with a batch size of 16.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Keras for building and training the deep learning model.
- OpenCV for image processing.
- Matplotlib and Seaborn for visualizations.
- Tkinter for the GUI interface.
