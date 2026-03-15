# Sensor Calibration using AI

This repository contains the code and for the Deep Learning Term-paper at University West. 

## Project Objective
Sensor data is a accuired from reality. If the true measurand is x, the sensor measures y = f(x). The goal of this project is to find the true x from this faulty reading through f^{-1}(y), which is the process of calibration. We chosed the "Air Pollution in Seoul" dataset to simulate sensor drift and designed various algorithms to correct it.

## Models Implemented
Following a spiral development approach, the calibration pipeline evolved from basic machine learning to deep learning models:
1. Linear Regression
2. Random Forest Regressor
3. Support Vector Regressor (SVR)
4. Recurrent Neural Network (RNN)

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deep Learning:** TensorFlow / Keras
* **Visualization:** Matplotlib, Seaborn
