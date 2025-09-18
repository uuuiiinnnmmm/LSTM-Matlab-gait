# LSTM-Based Gait Analysis using IMU Data in MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-orange.svg)
![Toolbox](https://img.shields.io/badge/Toolbox-Deep%20Learning-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> A robust and modular framework for Inertial Measurement Unit (IMU) based gait analysis using Long Short-Term Memory (LSTM) networks, implemented with the MATLAB Deep Learning Toolbox. This project provides a clear workflow for preprocessing time-series sensor data and training a deep learning model for classification tasks.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [✨ Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🛠️ System Requirements](#️-system-requirements)
- [🚀 Getting Started: A Step-by-Step Guide](#-getting-started-a-step-by-step-guide)
- [⚙️ The Workflow in Detail](#️-the-workflow-in-detail)
- [📊 Results](#-results)
- [🤝 How to Contribute](#-how-to-contribute)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 📖 Overview

Human gait analysis is a critical tool in clinical diagnostics, rehabilitation monitoring, and biometric identification. This project leverages the power of LSTM networks, which are exceptionally well-suited for learning from sequential data, to classify gait patterns from 6-axis IMU sensors (3-axis accelerometer and 3-axis gyroscope).

The entire pipeline, from data ingestion to model evaluation, is built within the MATLAB environment, ensuring a cohesive and reproducible research platform. The primary objective is to provide a clean, well-documented, and easily adaptable template for researchers and engineers tackling similar time-series classification problems.

## ✨ Features

This framework is built with clarity and best practices in mind:

- **🔌 Modular Workflow**: Code is cleanly separated into `preprocessing` and `training` scripts for better management, readability, and reusability.
- **📁 Direct `.mat` File Handling**: Efficiently loads and processes standard MATLAB `.mat` data files without requiring intermediate format conversions.
- **✂️ Sliding Window Segmentation**: Implements a robust sliding window technique to segment continuous time-series data into fixed-length samples suitable for LSTM models.
- **🛡️ Data Leakage Prevention**: Applies Z-score normalization based *only* on the training set statistics to ensure the model generalizes well to unseen data.
- **🧠 Stacked LSTM Architecture**: Demonstrates the implementation of a multi-layered (stacked) LSTM network to capture hierarchical temporal features effectively.
- **📈 Comprehensive Evaluation**: Includes model evaluation with accuracy metrics and a detailed confusion matrix visualization to assess classification performance.

## 📂 Repository Structure

The repository is organized to maintain a clear separation of concerns between code, data, and results.

```bash
gait-analysis-lstm/
├── .gitignore          # Specifies files and folders to be ignored by Git
├── LICENSE             # The MIT License file
├── README.md           # This readme file
├── requirements.txt    # Lists required MATLAB version and toolboxes
├── data/
│   └── RawData/
│       └── README.md   # Instructions for placing and formatting raw data
├── results/
│   └── README.md       # Directory for saving output figures and models
└── scripts/
    ├── preprocess_gait_data.m  # Script for all data loading and preprocessing
    └── train_gait_lstm.m       # Script for building, training, and evaluating the model
```

## 🛠️ System Requirements

To successfully run this project, your environment must meet the following requirements:

- **MATLAB Version: R2022a or newer**

- **Required MATLAB Toolboxes:**

  - Deep Learning Toolbox™

  - Statistics and Machine Learning Toolbox™

  - (Optional) Signal Processing Toolbox™

A detailed list of dependencies can be found in the requirements.txt file.

## 🚀 Getting Started: A Step-by-Step Guide

Follow these clear steps to get the project running on your local machine.

**Step 1: Clone the Repository**

Open your terminal and clone this repository to your desired location.

```bash
git clone https://github.com/YourUsername/gait-analysis-lstm.git
cd gait-analysis-lstm
```
  

**Step 2: Add Your Raw Data**

1.Navigate to the data/RawData/ directory.

2.Place your raw .mat files inside this folder.

3.Crucial: Ensure each .mat file contains a single variable named data with dimensions N x D (N = time points, D = features). The filename (e.g., Healthy.mat, PatientA.mat) will be automatically used as the class label.

**Step 3: Run the Preprocessing Script**

1.Open MATLAB.

2.Navigate the MATLAB environment to the project's root directory (gait-analysis-lstm/).

3.Execute the preprocessing script from the MATLAB Command Window:
  ```bash
    run('scripts/preprocess_gait_data.m');
```
      

This will create a new file, data/preprocessed_gait_data.mat, containing the structured training and testing datasets.

**Step 4: Run the Training Script**

Once preprocessing is complete, execute the model training script in the same way:
```bash   
    run('scripts/train_gait_lstm.m');
```
      

This will initialize the training process. A real-time training progress plot will appear. Upon completion, the final test accuracy and a confusion matrix figure will be generated.

## ⚙️ The Workflow in Detail

The project pipeline is executed in two primary stages:
**1. Data Preprocessing (preprocess_gait_data.m)**

- Load: Ingests all .mat files from the data/RawData/ directory.

- Window: Segments continuous data streams into overlapping windows of a predefined size (e.g., 128 time steps).

- Label: Assigns a class label to each window based on its source filename.

- Partition: Splits the data into training (70%) and testing (30%) sets while maintaining class distribution.

- Normalize: Calculates the mean and standard deviation from the training data and applies Z-score normalization to both training and testing sets.

- Save: Saves the processed, partitioned, and normalized data into a single preprocessed_gait_data.mat file.

**2. Model Training & Evaluation (train_gait_lstm.m)**

- Load: Loads the preprocessed_gait_data.mat file.

- Define Architecture: Defines a stacked LSTM network, including sequence input, LSTM, dropout, fully connected, softmax, and classification layers.

- Train: Trains the network using the Adam optimizer with specified hyperparameters (epochs, learning rate, etc.).

- Evaluate: Assesses the trained model's performance on the unseen test set and reports the final classification accuracy.

- Visualize: Generates and displays a confusion matrix to provide detailed insights into the model's performance across different classes.

## 📊 Results

After successfully running the training script, the model's performance metrics will be available. The expected outputs are:

1.Final Test Accuracy: Printed directly in the MATLAB Command Window.

2.Confusion Matrix: A new figure window displaying the classification results for the test set, allowing for a detailed error analysis.

You can save the generated figures and trained model net object in the results/ folder for future reference.

You can insert an example image of your results here:
```bash
<!-- ![Confusion Matrix Example](results/confusion_matrix.png) -->
```

## 🙏 Acknowledgments

- This project was inspired by the need for a clear and reproducible workflow in academic research for time-series classification.

- Thanks to the MathWorks community for their excellent documentation on the Deep Learning Toolbox.
