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

## Getting Started



