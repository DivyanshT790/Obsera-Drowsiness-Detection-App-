# OBSERA — Intelligent Driver Drowsiness Detection System

> **OBSERA** is a real-time computer-vision-based driver drowsiness detection system that analyzes eye behavior, mouth movement, and head pose to identify signs of driver fatigue and drowsiness.

---

## 📌 Overview

Driver fatigue is one of the major causes of road accidents. A driver may become drowsy gradually, making it difficult to recognize the danger before a critical event occurs.

**OBSERA** addresses this problem by continuously analyzing a driver's facial behavior through a camera.

Instead of depending on a single indicator such as closed eyes, OBSERA combines multiple signals:

- 👁️ **Eye behavior**
- 😮 **Mouth/yawning behavior**
- 🧭 **Head orientation**
- 📊 **Temporal drowsiness patterns**

These signals are processed and combined to generate a final drowsiness assessment.

### Core Idea

```text
                Camera
                  │
                  ▼
           Video Frame Input
                  │
                  ▼
        Face & Landmark Detection
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Eye Analysis  Mouth      Head Pose
       │         Analysis      │
       │            │           │
       ▼            ▼           ▼
   CNN + EAR     SVM + MAR   Pitch/Yaw/Roll
       │            │           │
       └────────────┼───────────┘
                    ▼
             Feature Fusion
                    │
                    ▼
          Drowsiness Score
                    │
                    ▼
       Alert / Normal Condition
```

---

# 🎯 Objectives

OBSERA was designed with the following objectives:

1. Detect early signs of driver drowsiness.
2. Analyze eye closure and eye behavior.
3. Detect yawning and prolonged mouth opening.
4. Monitor abnormal head orientation.
5. Combine multiple indicators instead of relying on one measurement.
6. Provide real-time feedback.
7. Build a lightweight computer-vision-based solution that can operate using a standard camera.

---

# 🧠 How OBSERA Works

OBSERA uses a **hybrid computer vision and machine learning architecture**.

The system does not rely entirely on one neural network.

Instead, it combines:

- **MediaPipe Face Mesh** for facial landmarks
- **EAR** for eye geometry
- **CNN** for eye-state classification
- **MAR** for mouth geometry
- **SVM** for mouth/yawn classification
- **Head-pose estimation** for facial orientation
- **Decision/fusion logic** for the final drowsiness assessment

This approach provides multiple independent sources of evidence.

---

# 🔍 System Workflow

## 1. Video Capture

The system receives live frames from a camera.

```text
Camera
  ↓
Video Stream
  ↓
Frame-by-frame Analysis
```

OpenCV is used for image and video processing.

---

## 2. Facial Landmark Detection

OBSERA uses **MediaPipe Face Mesh** to identify important facial landmarks.

These landmarks provide the coordinates required for geometric measurements.

Important regions include:

- Left eye
- Right eye
- Mouth
- Face orientation points

The landmarks are then used to calculate eye and mouth measurements.

---

# 👁️ Eye Drowsiness Detection

Eye analysis uses **two complementary approaches**:

### 1. EAR — Eye Aspect Ratio

EAR measures how open or closed the eye is.

A simplified formulation is:

\[
EAR =
\frac{\|p_2-p_6\|+\|p_3-p_5\|}
{2\|p_1-p_4\|}
\]

Where the points represent specific eye landmarks.

When the eye is open:

```text
     ______
   /        \
  |    ●     |
   \________/
```

The vertical distance is relatively large.

When the eye closes:

```text
   __________
```

The vertical distance becomes small.

Therefore, a sustained decrease in EAR can indicate eye closure.

### Why EAR?

EAR is:

- Computationally inexpensive
- Based on facial geometry
- Easy to calculate in real time
- Useful for detecting prolonged eye closure

However, EAR alone can be affected by factors such as individual eye shape, camera angle, and landmark noise.

---

# 🧠 CNN-Based Eye Classification

OBSERA also uses a **Convolutional Neural Network (CNN)** for eye-state recognition.

The CNN receives visual information from the eye region and learns patterns associated with different eye states.

```text
Eye Image
    ↓
Convolution Layers
    ↓
Feature Extraction
    ↓
Pooling
    ↓
Dense Layers
    ↓
Eye Classification
```

### Why CNN for Eyes?

The eye region contains rich visual information.

A CNN can learn spatial features such as:

- Eyelid position
- Eye shape
- Visual appearance of closed eyes
- Local image patterns

Therefore, CNN-based classification complements the geometric EAR measurement.

### Hybrid Eye Analysis

```text
                Eye Region
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
        CNN                  EAR
          │                   │
          ▼                   ▼
   Visual Eye State      Eye Geometry
          │                   │
          └─────────┬─────────┘
                    ▼
              Eye Evidence
```

This allows OBSERA to use both **learned visual information** and **geometric information**.

---

# 😮 Mouth and Yawning Detection

Yawning is another important indicator of driver fatigue.

OBSERA uses **MAR — Mouth Aspect Ratio** to measure mouth opening.

A simplified representation is:

\[
MAR =
\frac{\text{vertical mouth distances}}
{\text{horizontal mouth distance}}
\]

When the mouth opens widely, MAR increases.

However, a large MAR value alone does not necessarily mean yawning. A driver could simply be talking.

Therefore, OBSERA uses an **SVM classifier** alongside MAR.

---

# 🤖 SVM-Based Yawning Classification

The mouth landmarks are converted into numerical features such as:

```text
MAR
Mouth geometry
Mouth opening behavior
```

These features are provided to a **Support Vector Machine (SVM)**.

```text
Mouth Landmarks
       ↓
      MAR
       ↓
Feature Vector
       ↓
      SVM
       ↓
Normal / Yawning
```

### Why SVM for Mouth Analysis?

The mouth analysis primarily uses **compact numerical/geometric features** rather than large image representations.

SVM is suitable because it:

- Works well with low-dimensional features
- Can perform well with relatively small datasets
- Provides an efficient classification approach
- Avoids the computational cost of training another deep CNN

Therefore:

> **CNN is used where visual spatial information is important, while SVM is used where compact engineered features such as MAR are sufficient.**

---

# 🧭 Head Pose Estimation

Eye and mouth behavior alone may not provide enough information.

OBSERA also estimates the orientation of the driver's head.

Three primary angles are considered:

### Pitch

Up/down movement of the head.

```text
      ↑
    Pitch
      ↓
```

### Yaw

Left/right rotation.

```text
←   Face   →
     Yaw
```

### Roll

Tilting the head toward either shoulder.

```text
   / Face
  /
Roll
```

These measurements can help identify abnormal head orientation associated with fatigue or loss of attention.

---

# 📊 Drowsiness Decision System

OBSERA combines multiple signals rather than making a decision from a single measurement.

Conceptually:

```text
             Eye State
                 │
                 ▼
             EAR/CNN
                 │
                 │
Yawning ───► MAR/SVM
                 │
                 │
Head Pose ─► Pitch/Yaw/Roll
                 │
                 ▼
          Feature Fusion
                 │
                 ▼
       Drowsiness Assessment
                 │
          ┌──────┴──────┐
          ▼             ▼
        Normal       Drowsy
```

The system uses a scoring/fusion mechanism to combine the available evidence.

This makes the system more robust than a simple rule such as:

```text
IF eyes closed → Drowsy
```

---

# 🏗️ System Architecture

```text
                         ┌───────────────────┐
                         │      Camera       │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   OpenCV Frame    │
                         │     Processing    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ MediaPipe Face    │
                         │      Mesh         │
                         └─────────┬─────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
          ┌────────────┐    ┌────────────┐    ┌────────────┐
          │    Eyes    │    │   Mouth    │    │ Head Pose  │
          └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
                │                 │                 │
          ┌─────┴─────┐      ┌────┴────┐      ┌────┴────────┐
          │           │      │         │      │             │
          ▼           ▼      ▼         ▼      ▼             ▼
         EAR         CNN    MAR       SVM   Pitch         Yaw/Roll
          │           │      │         │      │             │
          └─────┬─────┘      └────┬────┘      └──────┬──────┘
                │                 │                  │
                └─────────────────┼──────────────────┘
                                  ▼
                         ┌───────────────────┐
                         │ Feature Fusion /  │
                         │ Drowsiness Score  │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Alert / Status   │
                         └───────────────────┘
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **OpenCV** | Camera input and image processing |
| **MediaPipe** | Facial landmark detection |
| **TensorFlow / Keras** | CNN eye model |
| **Scikit-learn** | SVM-based mouth/yawn classifier |
| **Joblib** | Loading the trained SVM model |
| **NumPy** | Numerical computations |
| **Pandas** | Data processing |
| **Streamlit** | Web application interface |
| **Streamlit-WebRTC** | Real-time camera/video streaming |

---

# 📁 Project Structure

A typical OBSERA project structure is:

```text
OBSERA/
│
├── app/
│   └── ...
│
├── models/
│   ├── eye_cnn_premade.h5
│   └── yawn_svm.joblib
│
├── assets/
│   └── ...
│
├── requirements.txt
├── README.md
└── app.py
```

> The exact structure may differ depending on the deployment version of the project.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/OBSERA.git
cd OBSERA
```

## 2. Create a Virtual Environment

It is recommended to use **Python 3.10–3.12** for compatibility with the computer-vision and machine-learning dependencies.

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running OBSERA

For the Streamlit version:

```bash
streamlit run app.py
```

The application will start a local web server and provide the interface for real-time drowsiness detection.

---

# 📋 Model Files

OBSERA uses two primary trained models.

### Eye CNN

```text
models/eye_cnn_premade.h5
```

Used for eye-state classification.

### Yawn SVM

```text
models/yawn_svm.joblib
```

Used for mouth/yawning classification.

The models should be present at the expected paths before starting the application.

---

# 🔬 Machine Learning Approach

OBSERA follows a **hybrid ML + computer vision approach**.

| Signal | Feature | Model/Method | Purpose |
|---|---|---|---|
| Eye | EAR | Geometric analysis | Measure eye closure |
| Eye | Eye image | CNN | Classify eye state |
| Mouth | MAR | Geometric analysis | Measure mouth opening |
| Mouth | MAR/features | SVM | Classify yawning |
| Head | Facial landmarks | Pose estimation | Detect head orientation |
| Final | Multiple signals | Fusion/scoring | Determine drowsiness |

---

# 💡 Why a Hybrid Approach?

A purely deep-learning-based solution is not always necessary for every part of the problem.

OBSERA separates the problem into different components based on the nature of the information.

### For eyes

Visual appearance matters significantly.

Therefore:

```text
Eye Image → CNN
```

At the same time, geometric eye closure can be measured efficiently:

```text
Eye Landmarks → EAR
```

### For mouth

The primary information can be represented compactly using geometry:

```text
Mouth Landmarks → MAR → SVM
```

### For head

The required information is primarily geometric:

```text
Facial Landmarks → Head Pose
```

This results in a **hybrid architecture combining deep learning, classical machine learning, and geometric computer vision**.

---

# ⚠️ Limitations

OBSERA is a research/academic prototype and should not be considered a certified automotive safety system.

Performance can be affected by:

- Poor lighting
- Camera placement
- Extreme head rotations
- Occlusion
- Sunglasses
- Face masks
- Low camera resolution
- Facial landmark detection errors
- Individual differences in facial features
- False positives from talking, laughing, or normal facial movements

The system should therefore be evaluated on a diverse dataset and under realistic driving conditions before any safety-critical deployment.

---

# 🚀 Future Improvements

Possible future improvements include:

### 1. Temporal Deep Learning

Instead of analyzing individual frames independently, temporal models such as:

- LSTM
- GRU
- Temporal CNN
- Transformer-based architectures

could model drowsiness patterns over time.

### 2. Better Multimodal Fusion

A learned fusion model could combine:

```text
EAR
MAR
CNN eye probability
SVM yawning probability
Head pose
Blink duration
Eye closure duration
```

instead of relying only on manually defined weights.

### 3. Personalized Calibration

Different people have different natural EAR and MAR values.

An adaptive calibration stage could establish a driver's baseline measurements.

### 4. Improved Robustness

The system could be trained and tested under:

- Daylight
- Night conditions
- Glasses
- Different camera positions
- Different ethnicities and facial structures
- Different vehicle interiors

### 5. Edge Deployment

OBSERA could eventually be optimized for:

- NVIDIA Jetson
- Raspberry Pi
- Android
- Embedded automotive systems

for real-time in-vehicle deployment.

---

# 📈 Evaluation Metrics

For proper model evaluation, the following metrics can be used:

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Real-Time System Metrics

- FPS
- Inference latency
- CPU/GPU utilization
- Memory consumption

For a safety-oriented application, **recall for the drowsy class is particularly important**, because missing a genuinely drowsy driver can be more consequential than generating an occasional false alert.

---

# 🔐 Privacy

OBSERA is designed around real-time camera analysis.

For a production deployment, the system should preferably process video locally rather than uploading raw camera footage to a remote server.

Any stored video or facial data should be handled according to applicable privacy and data-protection requirements.

---

# 📌 Key Features

- Real-time facial analysis
- Eye-state detection
- EAR calculation
- CNN-based eye classification
- MAR calculation
- SVM-based yawning detection
- Head-pose estimation
- Multi-signal drowsiness assessment
- Streamlit interface
- WebRTC-based video streaming
- Modular ML architecture

---

# 🎓 Project Significance

OBSERA demonstrates how **computer vision, machine learning, facial landmark analysis, and real-time streaming** can be combined to address a practical safety problem.

Rather than treating drowsiness as a single classification problem, the system decomposes it into measurable behavioral indicators and combines them into a unified assessment.

The project therefore provides practical experience in:

- Computer Vision
- Machine Learning
- Deep Learning
- Feature Engineering
- Facial Landmark Analysis
- Real-Time Video Processing
- Model Integration
- Web Application Deployment

---

# 👨‍💻 Author

**Divyansh Tiwari**

B.Tech — Information Technology  
BBD NIIT, Lucknow

---

# 📄 License

This project is intended for **educational and research purposes**.

Add an appropriate open-source license such as MIT if you intend to allow others to modify and redistribute the project.

---

## ⭐ Project Summary

**OBSERA** is a hybrid driver-drowsiness detection system that combines **MediaPipe facial landmarks, EAR, MAR, CNN-based eye classification, SVM-based yawning classification, and head-pose estimation** to identify multiple behavioral indicators of driver fatigue.

The central idea is simple:

> **Don't depend on one signal to determine drowsiness. Combine independent visual and geometric indicators to make the assessment more reliable.**
