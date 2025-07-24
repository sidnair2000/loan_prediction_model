# Loan Approval Prediction Using Machine Learning

## Project Overview

This project applies five different machine learning algorithms to predict whether a loan should be approved for a customer based on their personal and financial data. The goal is to compare these models' performances and select the best one for accurate and efficient loan approval prediction.

## Models Used

- Logistic Regression  
- Support Vector Classifier (SVC)  
- Keras Multi-Layer Perceptron (MLP)  
- Deep Dense Neural Network with Batch Normalization  
- Simple LSTM  

## Dataset

The dataset includes customer details relevant to loan approval, such as income, loan amount, credit history, and more. Identifiers like `Loan_ID` are dropped before modeling as they do not contribute to prediction.

## Data Preprocessing

- Dropped irrelevant columns (`Loan_ID`) and saved test IDs for result mapping.  
- Checked for missing values and imputed them using mode for categorical features and median for numerical features.  
- Applied log transformation on skewed features (e.g., `LoanAmount`) to reduce skewness and mitigate outliers.  
- Encoded categorical variables using `LabelEncoder`.  
- Split data into training and validation sets (75% train, 25% validation).  
- Normalized feature data to prepare for model input.

## Model Training and Evaluation

Each model was trained on the training set and evaluated on the validation set using accuracy and weighted F1 score. Confusion matrix heatmaps were generated for visual performance comparison.

- **Logistic Regression:** Baseline model using simple linear decision boundaries.  
- **SVC:** Support Vector Classifier with good accuracy and relatively low computational cost.  
- **Keras MLP:** Neural network with 3 dense layers, using ReLU activations and sigmoid output for binary classification.  
- **Deep Dense NN with Batch Normalization:** Improved deep neural network adding batch normalization layers for better training stability.  
- **Simple LSTM:** Recurrent neural network model processing data sequences with one LSTM layer.

## Results

A summary DataFrame compiles each model’s accuracy and F1 score for easy comparison.

## How to Run

1. Install required packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `seaborn`, and `matplotlib`.  
2. Prepare your dataset by placing the CSV files in the working directory.  
3. Run the notebook/script step-by-step, starting from data preprocessing to model training and evaluation.  
4. Review the summary output for model performance and visualize confusion matrices for deeper insights.

## Notes

- Random seeds are fixed to improve reproducibility.  
- Heatmaps visualize confusion matrices to help understand model prediction strengths and weaknesses.  
- The final model selection balances accuracy, F1 score, and computational efficiency.
