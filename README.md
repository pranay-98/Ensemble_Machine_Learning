# Ensemble_Machine_Learning
All State Insurance Claims Severity Prediction

## Introduction

This project is aimed at performing exploratory data analysis (EDA) on the Allstate insurance claim dataset and building machine learning models to predict the claim amount. The objective is to understand the data distribution, handle missing values, perform feature selection and elimination, and ultimately build and deploy a machine learning model.

## Data Analysis

The initial step of the project involved performing EDA on the Allstate insurance claim dataset. This included analyzing and studying the data distribution for categorical variables. Missing values were handled for both categorical and continuous variables, and outliers were treated using visual techniques like box plots.

## Encoding Techniques

Label and One-Hot-Encoder techniques were compared and analyzed to understand when to use each. The best encoding technique was used based on the results.

## Feature Selection and Elimination

Correlation, constant variance, and chi-square statistical tests were used for feature selection and elimination. This helped in reducing the number of features and improving the model's accuracy.

## Machine Learning Models

Random Forest and GBM regression ensemble machine learning models were built. The model accuracy for the random forest model was 68%, which was improved to 80% after hyperparameter tuning. The model accuracy for the GBM model was 72%. Hyperparameter tuning was performed using Sklearn functions, and the model selection was done using the RMSE metric as the evaluation criterion.

## Deployment

The final step involved deploying the machine learning model using Flask API. This API can be used to make predictions on new data.

## Monitoring and Experiment Tracking

ML Monitoring and Experiment Tracking was done using MLFoundry to track the experiments and monitor the performance of the models.

## Conclusion

In conclusion, this project involved performing EDA on the Allstate insurance claim dataset and building machine learning models to predict the claim amount. The objective was achieved by handling missing values, performing feature selection and elimination, and deploying the final model using Flask API. The model accuracy was improved through hyperparameter tuning and monitoring was done using MLFoundry.




