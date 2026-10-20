# Guitar Note Classifier

## Overview

The **Guitar Note Classifier** is a machine learning project designed to classify guitar notes from audio files. The system processes `.wav` and `.mp3` files, detects note onsets, converts them into Mel spectrograms, and uses a trained **Convolutional Neural Network (CNN)** to classify these notes.

This **README** provides a set up, train, and run the model on GitHub, ensuring a seamless workflow.

## Features

- Detects note onsets in an audio file using advanced onset detection techniques.
- Converts audio segments into Mel spectrograms for classification.
- Trains a CNN model to classify guitar notes.
- Applies Non-Maximum Suppression (NMS) to eliminate duplicate note detections.
- Interactive Plot (zoomable & hoverable with Plotly).

## Prerequisites

Ensure you have the following installed:

### Required Software
- Python 3.8 or later

### Required Python Libraries
Install them using:
```sh
pip install tensorflow librosa matplotlib numpy scikit-learn plotly
```

## Setup

### Clone the Repository
```sh
git clone <repository_url>
cd <repository_directory>
```

### Directory Structure
```
Guitar Dataset/
├── A2/
│   ├── file1.wav
│   ├── file2.wav
├── B3/
│   ├── file3.wav
│   ├── file4.wav
...
```

This structure ensures the script correctly processes files from Guitar Dataset for training and testing.

## Model Training (Automated on GitHub)

### Setup GitHub Actions

#### Create a `requirements.txt` file with dependencies:
```
tensorflow
librosa
matplotlib
numpy
scikit-learn
plotly
```

#### Create `.github/workflows/train_model.yml` with:
```yaml
name: Train Guitar Note Classifier

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Train the model
        run: python train_model.py
      
      - name: Save model artifact
        uses: actions/upload-artifact@v3
        with:
          name: note_classifier
          path: note_classifier.h5
```

## Training Process

1. **Load and preprocess audio data**: Convert `.wav` files into Mel spectrograms.
2. **Split the dataset**: 80% training, 20% testing.
3. **Build a CNN model**: Includes convolutional layers, max pooling, dropout, and dense layers.
4. **Compile the model**: Uses **Adam optimizer** and **categorical cross-entropy loss function**.
5. **Train the model**: Runs for **20 epochs** with **batch size 32**.
6. **Evaluate the model**: Assesses accuracy using the test dataset.
7. **Save the model**: Uploads `note_classifier.h5` as a GitHub artifact.

## Note Detection in a Musical Piece

### Run the Note Detection Script

Once trained, detect notes in an audio file using:
```sh
python detect_notes.py
```

### Process Overview

1. Loads an input musical piece (`.mp3` or `.wav`).
2. Detects **onsets** (where a note starts).
3. Applies **Non-Maximum Suppression (NMS)** to remove duplicate detections.
4. Converts detected segments into **Mel spectrograms**.
5. Uses the trained **CNN model** to classify the notes.
6. **Visualizes** the detected notes in different ways.

## Adjusting Parameters

Modify `detect_notes.py`:

| Parameter         | Description                   | Default        |
|------------------|------------------------------|---------------|
| `MUSICAL_PIECE_PATH` | Path to the audio file        | `'test_song.wav'` |
| `SEGMENT_DURATION`   | Duration of each note segment | `0.5s`        |
| `SR`               | Sampling rate                 | `44100 Hz`    |
| `THRESHOLD_NMS`    | Minimum gap between detected notes | `0.5s` |

## Outputs & Visualizations

### **Trained Model**
- Saved as **`note_classifier.h5`**.

### **Evaluation Metrics**
- Accuracy and loss plots.
- Confusion matrix and classification report.

### **Detected Notes**
Displayed in the terminal as:
```
Time: 1.23s -> Note: A3
Time: 2.45s -> Note: C#4
```

### **Visualization Options**
- **Interactive Zoomable Plot**

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No Notes Detected** | Ensure input audio has clear notes. Adjust onset detection parameters in `detect_notes.py`. |
| **Too Many Repeated Notes** | Adjust `THRESHOLD_NMS` to remove close detections. |
| **Training Issues** | Verify dataset structure (`Guitar Dataset/`). Ensure enough samples exist for each note. |

## Future Improvements

- Support for **more audio formats** (`.flac`, `.ogg`, etc.).
- More **advanced CNN architectures** (e.g., ResNet).
- **Real-time detection** & GUI interface.

## Contributions

This project was completed under the guidance of **Mr. Sreesha Kudoor**.

## Final Thoughts

This project provides an **end-to-end solution** for detecting and classifying guitar notes from audio files.

### **Next Steps**
✅ Train the model using **GitHub Actions**.  
✅ Test it on a song with **`detect_notes.py`**.  
✅ Fine-tune parameters for best accuracy.  

