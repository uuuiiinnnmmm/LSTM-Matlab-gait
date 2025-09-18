# LSTM-Based Gait Analysis using IMU Data in MATLAB

![MATLAB](https://img.shields.io/badge/MATLAB-R2023a%2B-orange.svg)
![Toolbox](https://img.shields.io/badge/Toolbox-Deep%20Learning-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A robust framework for Inertial Measurement Unit (IMU) based gait analysis using Long Short-Term Memory (LSTM) networks, implemented with MATLAB's Deep Learning Toolbox. This project provides a clear and modular workflow for preprocessing time-series sensor data and training a deep learning model for gait classification tasks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation and Usage Guide](#installation-and-usage-guide)
- [Workflow Details](#workflow-details)
- [Results](#results)
- [How to Contribute](#how-to-contribute)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Human gait analysis is a critical tool in clinical diagnostics, rehabilitation monitoring, and biometric identification. This project leverages the power of LSTM networks, which are exceptionally well-suited for learning from sequential data, to classify gait patterns from 6-axis IMU sensors (3-axis accelerometer and 3-axis gyroscope). The entire pipeline, from data ingestion to model evaluation, is built using MATLAB, ensuring a cohesive and reproducible research environment.

The primary objective is to provide a clean, well-documented, and easily adaptable template for researchers and engineers working on similar time-series classification problems.

## Features

- **Modular & Separated Workflow**: Code is cleanly separated into preprocessing and training scripts for better management and reusability.
- **Direct `.mat` File Handling**: Efficiently loads and processes standard MATLAB `.mat` data files without intermediate format conversions.
- **Sliding Window Segmentation**: Implements a robust sliding window technique to segment continuous time-series data into fixed-length samples suitable for LSTM models.
- **Training-Set Based Normalization**: Applies Z-score normalization based *only* on the training set statistics to prevent data leakage and ensure model generalization.
- **Stacked LSTM Architecture**: Demonstrates the implementation of a multi-layered (stacked) LSTM network to capture hierarchical temporal features.
- **Comprehensive Evaluation**: Includes model evaluation with accuracy metrics and visualization of a confusion matrix to assess classification performance.

## Repository Structure

The repository is organized to maintain a clear separation of concerns between code, data, and results.
