# Loan Approval Prediction Using Machine Learning

## Project Overview

This project applies five different machine learning algorithms to predict whether a loan should be approved for a customer based on their personal and financial data. The objective is to compare the performance of these models and select the most accurate and efficient one for loan approval prediction.

## Models Used

- Logistic Regression  
- Support Vector Classifier (SVC)  
- Keras Multi-Layer Perceptron (MLP)  
- Deep Dense Neural Network with Batch Normalization  
- Simple LSTM  

## Dataset

The dataset contains customer details relevant to loan approval, including income, loan amount, credit history, and more. Irrelevant identifiers such as `Loan_ID` are dropped during preprocessing as they do not contribute to prediction.

## Data Preprocessing

- Dropped irrelevant columns (`Loan_ID`)  
- Handled missing values:  
  - Mode imputation for categorical features  
  - Median imputation for numerical features  
- Applied log transformation to skewed features (e.g., `LoanAmount`)  
- Encoded categorical variables using `LabelEncoder`  
- Split the data into:
  - 60% Training Set  
  - 20% Validation Set  
  - 20% Test Set  
- Normalized feature values to ensure consistent model input  

## Model Training and Evaluation

Each model is trained using the training set and evaluated on the validation and test sets using:

- Accuracy Score  
- Weighted F1 Score  
- Confusion Matrix Heatmaps for performance visualization

### Model Summary:

- Logistic Regression: Serves as a simple, interpretable baseline model  
- SVC: Performs well with smaller datasets and handles non-linear boundaries  
- Keras MLP: Fully connected neural network with ReLU activations and sigmoid output  
- Deep Dense NN + BatchNorm: Enhanced neural net using batch normalization for stable training  
- Simple LSTM: Recurrent neural network for sequence-like inputs, tested for its adaptability

## Results

A summary DataFrame is generated to compare accuracy and F1 scores of each model on validation and test data. Visualizations include heatmaps of confusion matrices.

## How to Run

1. Clone this repository or download the code files  
2. Install required libraries:  
   ```bash
   pip install numpy pandas scikit-learn tensorflow seaborn matplotlib
