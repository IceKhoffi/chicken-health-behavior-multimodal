# Chicken Health & Behavior Detection

## Overview

This repository is designed for the early detection of health issues and anomalous behaviors in chickens. By integrating **computer vision** (object detection and behavior analysis) and **vocalization analysis**, this project aims to provide comprehensive insights for proactive farm management, significantly reducing economic losses due to diseases in poultry farming.

## Why This Project?

This project was developed as a solution for **Datathon 2025: AI-Powered Business Insight & Action**. The core idea, which this project addresses, is as follows:

**Competition Goal:**
To enable its participants to build models that transform raw data into valuable information, thereby empowering better business decisions.

**Background:**
Beyond the competition, the poultry sector in Indonesia faces persistent and severe threats from disease outbreaks. Diseases like **Newcastle Disease (ND)** and **Chronic Respiratory Disease (CRD)** cause massive financial losses, with mortality rates reaching up to 100% for ND and 30% for complicated CRD. Manual detection methods employed by farmers are often inefficient, prone to human error due to fatigue, and lack the precision required for early intervention. This results in substantial direct losses (hundreds of millions to billions of rupiah annually per farm) and significant opportunity costs in unrealized production potential.

This project directly addresses this critical need by providing an intelligent monitoring system that leverages diverse data modalities to enhance the resilience and sustainability of the poultry industry, fulfilling the competition's objectives and offering real-world value.

## The Multimodal Solution

Our system integrates two primary AI approaches to provide a holistic view of chicken health and behavior:

1.  **Computer Vision:** Analyzing video and image data to detect individual chickens, track their movements, estimate density, and identify unusual visual cues related to illness or stress.
2.  **Vocalization Analysis:** Processing audio recordings of chicken sounds to classify vocalizations (e.g., healthy, unhealthy, noise) that can indicate respiratory issues or distress.

By combining insights from both vision and vocalization, the system aims to achieve a more robust and accurate early detection capability than single-modality approaches.

## Key Features and Components

* **Chicken Object Detection**
  *  Utilizes a **YOLOv11s** model to accurately detect and locate individual chickens in images and video frames.
  *  Forms the foundation for counting, density estimation, and individual tracking.
* **Chicken Vocalization Classification:**
  * Employs a **Convolutional Neural Network (CNN)** model to classify chicken vocalizations into `Healthy`, `Noise`, and `Unhealthy` categories.
  * Provides early indicators of respiratory diseases or general distress through sound analysis.

## Dataset & Models on Hugging Face

All data and model weights are publicly available on Hugging Face:

* **Dataset:**
  * **Chicken Health and Behavior Multimodal Dataset:** [`https://huggingface.co/datasets/IceKhoffi/chicken-health-behavior-multimodal`](https://huggingface.co/datasets/IceKhoffi/chicken-health-behavior-multimodal)
    * Contains image and video data for computer vision tasks, sourced from publicly available **Kipster Farm** YouTube videos.
    * Includes audio data for vocalization analysis, sourced from the "Poultry Vocalization Signal Dataset for Early Disease Detection" by Aworinde et al.

* **Models:**
  * **Chicken Object Detection Yolov11s:** [`https://huggingface.co/IceKhoffi/chicken-object-detection-yolov11s`](https://huggingface.co/IceKhoffi/chicken-object-detection-yolov11s)
  * **Chicken Vocalization Classifier:** [`https://huggingface.co/IceKhoffi/chicken-vocalization-classifier`](https://huggingface.co/IceKhoffi/chicken-vocalization-classifier)

## Getting Started (**Python 3.11.13**)

To set up the environment and explore the project : 

1. **Clone the repository:**
   ```
   !git clone https://github.com/your_username/chicken-health-behavior-multimodal.git
   !cd chicken-health-behavior-multimodal
   ```
2. **Install Dependencies:**
   * It is highly recommended to use this on `Google Collabs` or `virtual environment` (`conda` or `venv`).
   * Install necessary libraries. Refer to the `requirements.txt`
   ```
   !pip install -r requirements.txt
   ```
3. **Start Exploring**
   * You can navigate to the `/notebooks` directory. Each subfolder contains Jupyter Notebooks (`.ipynb`) guiding you through data preparation, model training, and evaluation for both vision and vocalization components.
   * Or if you want just to run the model you can refer to `main.ipynb` where it would run throught the chicken detection model to the vocalization model.

## Results
*(This section showcases key outputs and demonstrations from our models. For detailed code, full examples, and more analysis, please refer to the respective Jupyter notebooks in the `notebooks/` directory.)*

### Chicken Object Detection
*This output is from `notebooks/CHBD_TF_YOLOv11s_Distance_Estimation.ipynb`*

![Distance Estimation](https://github.com/user-attachments/assets/c489f2e9-8cdb-49ed-aa67-9c4e60da03a8)

*This output is from `notebooks/CHBD_TF_YOLOv11s_Density_Estimation.ipynb`*

![Density Estimation](https://github.com/user-attachments/assets/14357de7-fe37-41f6-85d8-3c844d32a66a)

### Chicken Vocalization Classification
*This output is from `notebooks/CHBD_Vocalization_Analysis.ipynb` of the performance metrics*

![Vocalization Classification Report and Confussion Matrix](https://github.com/user-attachments/assets/4c34ade6-ad39-4e68-8958-3d5ed47d8151)

*This output is from `notebooks/CHBD_Vocalization_Analysis.ipynb` of the test data prediction*

```
Predicted class: Healthy

Class Probabilities:
Healthy: 77.60%
Noise: 0.79%
Unhealthy: 21.60%
```

*To see the video results, you can refer to this Google Drive link : https://drive.google.com/drive/folders/1Z4ciNB1MprOILPV8Zn3VsEefEeuA2120?usp=sharing*


## Acknowledgements

* **Kipster Farm:** We extend our gratitude to Kipster Farm for making their informative videos publicly available on YouTube, which served as a crucial source for our visual dataset.
* **Poultry Vocalization Signal Dataset:** We gratefully acknowledge the creators of the "Poultry Vocalization Signal Dataset for Early Disease Detection" for their valuable contribution to the audio component of this project.
    * Aworinde, Halleluyah; Adebayo, Segun; Akinwunmi, Akinwale; Alabi, Olufemi; Ayandiji, Adebamiji; Oke, Olaide; Oyebamiji, Abel; Adeyemo, Adetoye; Sakpere, Aderonke; Echetama, Kizito (2023), “Poultry Vocalization Signal Dataset for Early Disease Detection”, Mendeley Data, V1, doi: [10.17632/zp4nf2dxbh.1](https://data.mendeley.com/datasets/zp4nf2dxbh/1)
* **Ultralytics YOLO:** This project extensively uses the Ultralytics YOLO framework.






