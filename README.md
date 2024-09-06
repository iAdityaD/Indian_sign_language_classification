
# Indian Sign Language Classification

## Overview
This project focuses on classifying Indian Sign Language (ISL) hand signs using computer vision techniques. It utilizes **Mediapipe** for hand tracking, **OpenCV** for real-time video capture, and **Python** for implementation. The model helps in detecting and recognizing ISL gestures, providing a significant step towards bridging communication gaps for the hearing and speech-impaired community.

## Features
- **Real-time hand tracking** using Mediapipe's Hand Landmark Model.
- **Gesture classification** of Indian Sign Language using a machine learning model.
- **User-friendly interface** for real-time video input through OpenCV.
- Customizable and scalable to include more hand gestures and signs.

## Tech Stack
- **Python**: Core programming language used for implementation.
- **Mediapipe**: Used for detecting hand landmarks in real-time.
- **OpenCV**: For capturing video feed and processing frames.
- **Machine Learning**: Trained classifier for gesture recognition.

## How It Works
1. **Hand Detection**: Mediapipe is used to detect and track hand landmarks in real-time through webcam input.
2. **Preprocessing**: The captured frames are processed to extract hand landmarks which are then converted into feature vectors.
3. **Classification**: The extracted features are fed into a pre-trained machine learning model that classifies the hand signs into different Indian Sign Language gestures.
4. **Display**: The recognized gestures are displayed on the screen along with the real-time video feed.

## Setup and Installation

### Prerequisites
- Python 3.x
- Mediapipe
- OpenCV
- Numpy
- Scikit-learn (for machine learning models)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/iAdityaD/Indian-Sign-Language-Classification.git
   cd Indian-Sign-Language-Classification
   ```

2. Install the required libraries:
   ```bash
   pip install mediapipe opencv-python numpy scikit-learn
   ```

3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

### Running the Project
1. Run the Python script:
   ```bash
   python main.py
   ```
2. The webcam will start, and hand gestures will be tracked in real-time. The classified Indian Sign Language gesture will be displayed on the screen.

## Data Collection and Training
- The dataset consists of images or video frames of various Indian Sign Language gestures.
- The hand landmarks are extracted using Mediapipe and fed into a machine learning classifier such as SVM, KNN, or any deep learning model.
- The model is trained on these feature vectors to classify the hand signs.

## Customization
- You can easily add more gestures by updating the dataset and retraining the model.
- Modify the `main.py` file to integrate more advanced machine learning models for better accuracy.

## Future Improvements
- Expand the dataset to include more ISL gestures.
- Improve classification accuracy with deep learning models such as Convolutional Neural Networks (CNNs).
- Add voice output for recognized gestures to make the system more interactive.

## Contributions
Feel free to fork this repository and contribute to the project by submitting a pull request.

## License
This project is licensed under the MIT License.
