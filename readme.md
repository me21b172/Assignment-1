# Assignment 1 Submission

## Overview
This submission contains the code and documentation for Assignment 1, including Python scripts, a link to the Wandb report, and the GitHub repository. The project focuses on training a neural network with various configurations and hyperparameter tuning.

## Code Organization
The code is organized into the following files:
1. **`train.py`**: 
   - Entry point for training the neural network.
   - Accepts command-line arguments for dataset, optimizer, learning rate, loss function, etc.
   - Performs Wandb sweeps and custom model fitting.

2. **`utilities.py`**: 
   - Contains functions for data preprocessing, stratified sampling, and model fitting.
   - Handles dataset extraction and normalization.

3. **`wandb_utils.py`**:
   - Includes Wandb integration for tracking experiments.
   - Implements hyperparameter sweep configurations and logging.

4. **`workhorse.py`**:
   - Implements the core neural network functionality.
   - Defines layers, forward propagation, backpropagation, optimizers, and gradient descent methods.

## Links
- **Wandb Report**: [Click here](https://wandb.ai/me21b172-indian-institute-of-technology-madras/NeuralNetwork-Hyperparameter-Tuning/reports/DA6401-Assignment-1--VmlldzoxMTY5ODU5Mg?accessToken=63cwykqc4090l1hkbjvg62af2yt9qeggl47lv0mpdopj4cxwma0iu9tgc948g925) to view the Wandb report.
- **GitHub Repository**: [Click here](https://github.com/me21b172/Assignment-1) to access the code repository.

## Instructions to Run
1. Extract the ZIP file into a directory.
2. Navigate to the directory in your terminal.
3. Install required Python libraries using:
