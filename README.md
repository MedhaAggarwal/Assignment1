# Course: CS502- Advanced Pattern Recognition

# Assignment-1: Logistic Regression on Titanic Dataset

## Overview
This assignment applies **Logistic Regression** on the famous Titanic dataset to predict passenger survival based on features like age, class, gender, and fare, etc.  

The dataset was downloaded from [Google Dataset Search](https://datasetsearch.research.google.com/) and contains passenger details along with survival status.

## Steps Implemented
1. Data Loading and Preprocessing
   - Filled missing values in `Age` and `Embarked`.
   - Encoded categorical variables (`Sex`, `Embarked`).
   - Standardized numerical features.

2. Model Training
   - Used **Logistic Regression**.
   - Split dataset into train (80%) and test (20%).
   - Evaluated using accuracy, confusion matrix, and classification report.

3. Visualization
   - Correlation heatmap to understand relationships among features.

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)

## Results
- Logistic Regression achieved **~80% accuracy** on the test dataset.
- Performance metrics were reported using classification report.

