# Ship Detection Using Machine Learning  

This project focuses on detecting ships in satellite imagery using machine learning models. It evaluates model performance in terms of accuracy, F1-score, and inference time and visually detects ships in large images using a sliding window approach.  

---

## Table of Contents  
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [How to Run](#how-to-run)  
6. [Outputs](#outputs)  
7. [Conclusion](#conclusion)  

---

## Introduction  

The project addresses maritime monitoring challenges using machine learning. Models are trained to classify satellite image patches as containing ships or not, and their performances are evaluated for operational use cases. The detection pipeline further identifies ships in large satellite images by processing them patch by patch.  

---

## Features  

- **Model Evaluation**: Compares various models like Random Forest and Linear Regression, with and without PCA.  
- **Inference Time Measurement**: Calculates the average time required for predictions.  
- **Ship Detection in Satellite Imagery**: Identifies and marks ships in large satellite images.  
- **Detailed Statistical Analysis**: Computes accuracy, F1-score, and inference time statistics.  

---

## Project Structure  

- **`evaluate_models.py`**: Script to evaluate machine learning models.  
- **`detect_ships.py`**: Script to detect ships in satellite images using a sliding window approach.  
- **Test Data**:  
  - `ship_data.npy`: Flattened test dataset.  
  - `ship_labels.npy`: Corresponding labels for the test dataset.  
- **Trained Models**:  
  - `Q1LinearRegression.pkl`: Linear Regression model.  
  - `Q1RandomForest.pkl`: Random Forest model.  
  - `Q3linearRegression.pkl`: PCA-based Linear Regression model.  
  - `Q3randomForest.pkl`: PCA-based Random Forest model.  
  - `Q4IsoMAp Random forest.pkl` : Iso Map based RF model.
  - `Q4IsoMap Linear Regression.pkl` : Iso map based linear regression

---

## Installation  

### Prerequisites  

Ensure you have Python 3.8+ installed. Install the required packages using:  
```bash
pip install numpy pandas scikit-learn matplotlib opencv-python joblib
```
## Outputs
Model Evaluation
The script outputs:

## Accuracy: How well the model predicts correct labels.
## F1-Score: A balance of precision and recall.
## Inference Time: Average time taken for predictions.

- ## Ship Detection
The satellite image is displayed with bounding boxes highlighting detected ships.

- ## Example Visualization
Satellite Image with Detected Ships
The detection algorithm scans the image in patches and highlights detected ships:

## Input Image: Large satellite image.
## Output: Same image with red bounding boxes indicating ship locations.
# Conclusion:
This project highlights the utility of machine learning models in real-world applications such as maritime surveillance. Advanced models like PCA-based Random Forest exhibit high accuracy but are computationally intensive. Conversely, simpler models offer quicker predictions, making them suitable for real-time scenarios. The sliding window detection approach further ensures robustness in identifying ships across varying image scales and resolutions.
