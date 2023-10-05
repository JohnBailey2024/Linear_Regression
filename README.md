# Simple Linear Regression

## Overview

This repository contains code and data for a simple and multiple linear regression analysis. 
## Contents

The repository includes the following files and directories:

1. `linear_regression.ipynb`: Jupyter Notebook containing the Python code for performing multiple linear regression analysis.
2. `credit.csv`: Sample dataset used for the analysis. It contains multiple independent variables (predictors) and one dependent variable.
3. `README.md`: This file, providing an overview and instructions for the project.

## Steps for Simple Linear Regression

Follow these steps to perform a simple linear regression analysis using the provided Jupyter Notebook (`linear_regression.ipynb`):

1. **Clone the Repository**:

   - Clone this repository to your local machine using Git:

     ```bash
     git clone https://github.com/John-Bailey2024/linear_regression.git
     cd linear_regression
     ```

2. **Install Required Libraries**:

   - Ensure you have Python (3.x recommended) installed on your system.
   - Install the necessary Python libraries by running:

     ```bash
      import matplotlib.pyplot as plt 
      import pandas as pd
      import numpy as np
      import statsmodels.api as sm
      import statsmodels.formula.api as smf
      from statsmodels.stats.outliers_influence import variance_inflation_factor
      from patsy import dmatrices
      import seaborn as sns
      from sklearn import linear_model
     ```

3. **Launch Jupyter Notebook**:

   - Start the Jupyter Notebook to access and run the analysis code:

     ```bash
     jupyter notebook linear_regression.ipynb
     ```

4. **Load and Explore Data**:
   - In the Jupyter Notebook, load your dataset or the provided sample dataset (`Credit.csv`).
   - Explore the dataset by examining its structure, summary statistics, and visualizations.

5. **Data Preprocessing**:

   - If necessary, preprocess the data, including handling missing values, outliers, and data transformations.

6. **Split Data into Training and Testing Sets**:

   - Split the dataset into a training set and a testing set to evaluate the model's performance.

7. **Fit the Linear Regression Model**:

   - Create a simple linear regression model using the scikit-learn library.
   - Train the model on the training data.

8. **Make Predictions**:

   - Use the trained model to make predictions on the testing data.

9. **Evaluate the Model**:

   - Assess the model's performance using appropriate evaluation metrics, such as mean squared error (MSE) or R-squared (R²).
   - Visualize the regression line and the scatterplot of the data points with the fitted line.

10. **Interpret the Results**:

    - Interpret the coefficients of the linear regression model to understand the relationship between the predictor and the dependent variable.

11. **Conclusion**:

    - Provide conclusions based on the analysis, including insights into the predictor's impact on the dependent variable.


# Multiple Linear Regression (MLR)

## Steps for Multiple Linear Regression

Follow these steps to perform a multiple linear regression analysis:

1. **Clone the Repository**:

   - Clone this repository to your local machine using Git (as mentioned in the Usage section).

2. **Install Required Libraries**:

   - Ensure you have Python and the necessary Python libraries installed (as mentioned in the Requirements section).

3. **Launch Jupyter Notebook**:

   - Start the Jupyter Notebook to access and run the analysis code (as mentioned in the Usage section).

4. **Load and Explore Data**:

   - In the Jupyter Notebook, load your dataset or the provided sample dataset (`Credit.csv`).
   - Explore the dataset by examining its structure, summary statistics, and visualizations.

5. **Data Preprocessing**:

   - If necessary, preprocess the data, including handling missing values, outliers, and data transformations.

6. **Split Data into Training and Testing Sets**:

   - Split the dataset into a training set and a testing set to evaluate the model's performance.

7. **Fit the Multiple Linear Regression Model**:

   - Create a multiple linear regression model using the scikit-learn library.
   - Train the model on the training data.

8. **Make Predictions**:

   - Use the trained model to make predictions on the testing data.

9. **Evaluate the Model**:

   - Assess the model's performance using appropriate evaluation metrics, such as mean squared error (MSE), R-squared (R²), and adjusted R-squared.

10. **Interpret the Results**:

    - Interpret the coefficients of the multiple linear regression model to understand the relationships between the predictors and the dependent variable.

11. **Conclusion**:

    - Provide conclusions based on the analysis, including insights into which predictors have a significant impact on the dependent variable.

## Dataset

`Credit.csv`


