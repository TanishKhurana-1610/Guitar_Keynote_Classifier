# Importing all the necessary libraries
import os # Handling the file path and directories
import numpy as np # For all the numerical operations
import librosa # To process audio 
import librosa.display
import tensorflow as tf # Developing machine Learning
from tensorflow.keras.models import load_model # To load the developed model
import matplotlib.pyplot as plt # To plot the spectogram
import plotly.graph_objects as go # To plot an interactive waveform

# Function to process the audio file and predict notes
def process_audio(file_path):
    MODEL_PATH = 'note_classifier.h5'  # Path to the trained model
    SR = 44100  # Sample rate
    TARGET_SHAPE = (128, 128)  # Input size for the spectrogram
    SEGMENT_DURATION = 0.5  # Segment duration in seconds

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Define musical notes
    classes = sorted([
        'A2', 'A3', 'A4', 'Asharp2', 'Asharp3', 'Asharp4',
        'B2', 'B3', 'B4', 'C3', 'C4', 'C5', 'Csharp3', 'Csharp4',
        'Csharp5', 'D2', 'D3', 'D4', 'D5', 'Dsharp2', 'Dsharp3',
        'Dsharp4', 'Dsharp5', 'E2', 'E3', 'E4', 'E5', 'F2', 'F3',
        'F4', 'F5', 'Fsharp2', 'Fsharp3', 'Fsharp4', 'Fsharp5',
        'G2', 'G3', 'G4', 'G5', 'Gsharp2', 'Gsharp3', 'Gsharp4', 'Gsharp5'
    ])

    # Load the audio file
    y, sr = librosa.load(file_path, sr=SR)
    print(f"Loaded music with {len(y)} samples at {sr} Hz.")

    # Detect note onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr) # Computing onset strength
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True) # Detecting onset frames (to detect the beging, where the notes begin)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr) # Converting onset frames into time values 

    # Non-Maximum Suppression (NMS) to remove duplicate detections
    def non_max_suppression(onsets, threshold=0.3):
        # To store the filtered onset time
        filtered_onsets = []
        #Loop through each detected note
        for i, onset in enumerate(onsets):
            if i == 0 or (onset - filtered_onsets[-1]) > threshold:
                filtered_onsets.append(onset)
                 #converting the list to Numpy array and return it
        return np.array(filtered_onsets)

    onset_times = non_max_suppression(onset_times, threshold=0.1)

    # List for detected notes
    predicted_notes = []
    
    # Iterating thorugh the onset times and processing the corresponding audio samples
    for onset_time in onset_times:
        # Converting the onset time into sample index
        onset_sample = int(onset_time * sr)
        # Defining sample length in samples
        segment_samples = int(SEGMENT_DURATION * sr)
        
        # Extract the segment of the waveform
        y_segment = y[onset_sample:onset_sample + segment_samples]
        
        # Pad with zeros if segment is too short
        if len(y_segment) < segment_samples:
            y_segment = np.pad(y_segment, (0, segment_samples - len(y_segment)), mode='constant')

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize spectrogram for model input
        mel_spec_db = tf.image.resize(np.expand_dims(mel_spec_db, axis=-1), TARGET_SHAPE)
        # Expand the dimensions to match the input format expected by the model
        input_data = np.expand_dims(mel_spec_db.numpy(), axis=0)

        # Predict musical note
        predictions = model.predict(input_data)
        # Only Keep the Highest Confidence Note
        predicted_index = np.argmax(predictions)
        # Retrieve the corresponding note label
        predicted_note = classes[predicted_index]

        # Store onset time and predicted note
        predicted_notes.append((onset_time, predicted_note))

    # Remove consecutive duplicate notes
    filtered_notes = []
    for i, (time, note) in enumerate(predicted_notes):
        if i == 0 or note != predicted_notes[i - 1][1]:
            filtered_notes.append((time, note))

    # Display detected notes
    print("\nDetected notes in the musical piece:")
    for time_val, note in filtered_notes:
        print(f"Time: {time_val:.2f}s -> Note: {note}")

    # Plot waveform with detected notes
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, len(y)/sr, num=len(y)), y=y, mode='lines', name='Waveform'))

    # Add vertical lines for detected notes
    for time_val, note in filtered_notes:
        fig.add_vline(x=time_val, line=dict(color="red", width=1, dash="dash"))
        fig.add_annotation(x=time_val, y=0.8, text=note, showarrow=True, arrowhead=2, font=dict(size=10, color="black"))

    # Customize layout
    fig.update_layout(
        title="Interactive Waveform with Detected Notes",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        showlegend=False
    )
    # To show the waveform
    fig.show()

# Get user input for file name
file_name = input("Enter the file name (with extension): ")
if os.path.exists(file_name):
    process_audio(file_name)
else:
    print("File not found. Please check the file name and try again.")



