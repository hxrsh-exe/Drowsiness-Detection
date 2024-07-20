
# Drowsiness Detection System

This project implements a drowsiness detection system using computer vision and deep learning techniques. The system monitors a user's eye state to detect signs of drowsiness and triggers an alarm if the user appears to be falling asleep.

## Project Components

The project consists of the following files:

1. **`alarm.wav`**: Audio file used for the alarm sound when drowsiness is detected.
2. **`drowsiness_detection.py`**: Python script for running the real-time drowsiness detection system using a webcam.
3. **`haar cascade files`**: Directory containing Haar cascade XML files used for face and eye detection.
4. **`model.py`**: Python script for defining, training, and saving the deep learning model used for drowsiness detection.
5. **`models`**: Directory where the trained model is saved.

## Prerequisites

- Python 3.x
- `pip` for installing Python packages

Install the necessary Python packages using the following commands:

`pip install -r requirements.txt`

## Setup and Usage

1.  **Place the Files**: Ensure that the following directory structure is followed:
    

    `Drowsiness Detection/`
    `├── alarm.wav`
    `├── drowsiness_detection.py`
    `├── haar_cascade_files/`
    `│   ├── haarcascade_frontalface_alt.xml`
    `│   ├── haarcascade_lefteye_2splits.xml`
    `│   └── haarcascade_righteye_2splits.xml`
    `├── model.py`
    `└── models/` 
    
2.  **Training the Model**:
    
    -   Navigate to the directory containing `model.py`.
    -   Ensure that the `data/train` and `data/valid` directories contain your training and validation images, respectively.
    -   Run the following command to train and save the model:

    `python model.py` 
    
    This will create a model file named `cnnCat2.h5` inside the `models` directory.
    
3.  **Running the Drowsiness Detection**:
    
    -   Ensure that you have a webcam connected and functioning.
    -   Run the drowsiness detection script:

    `python drowsiness_detection.py` 
    
    This script will open a window showing the webcam feed. If drowsiness is detected (i.e., both eyes are closed for a certain period), the alarm will sound.
    

## Notes

-   This project was developed on macOS. Ensure paths in the scripts are adjusted as per your file system structure.
-   The Haar cascade XML files and model weights should be correctly placed in their respective directories.
-   Modify file paths in `drowsiness_detection.py` if necessary to match your local setup.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Haar Cascade Classifiers: [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
-   Keras Documentation: [https://keras.io/](https://keras.io/)
-   OpenCV Documentation: https://docs.opencv.org/