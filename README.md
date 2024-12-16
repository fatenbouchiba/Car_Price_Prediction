# Car Price Prediction Using XGBoost

## Project Description

This project aims to predict the price of used cars based on various features using a machine learning model. The model is built using **XGBoost**, a gradient boosting framework that is known for its high performance on structured data. The project also implements **Explainable AI (XAI)** techniques, such as **SHAP** values, to interpret and explain the model's predictions.

## Features in the Dataset

The dataset consists of the following features:
- **Car_Name**: The name of the car
- **Year**: The year of manufacture
- **Selling_Price**: The price at which the car was sold (target variable)
- **Present_Price**: The current market price
- **Driven_kms**: The distance driven by the car (in kilometers)
- **Fuel_Type**: Type of fuel (Diesel, Petrol, etc.)
- **Selling_type**: Whether the selling type is 'Individual' or 'Dealer'
- **Transmission**: Whether the car has a manual or automatic transmission
- **Owner**: The number of previous owners

## Data Preprocessing

1. **Feature Engineering**: A new feature, `Car_Age`, is created by subtracting the `Year` from the current year.
2. **One-Hot Encoding**: Categorical features such as `Fuel_Type`, `Selling_type`, and `Transmission` are one-hot encoded.
3. **Scaling**: Numerical features like `Driven_kms` and `Present_Price` are scaled to have similar ranges.

## Model Selection and Hyperparameter Tuning

### Models Evaluated
- **Linear Regression**
- **Random Forest Regressor**
- **Support Vector Regressor (SVR)**
- **XGBoost Regressor** (chosen as the best model)

### Hyperparameters Optimized
- **Learning Rate**: 0.3
- **Max Depth**: 3
- **Number of Estimators**: 150

### Best Model
The **XGBoost** model was selected based on its **R² score** and **Mean Squared Error (MSE)**. It outperformed other models with an **R² score** of **0.8763** and **MSE** of **5.3225**.

## Model Evaluation

| Model                     | R² Score   | MSE    |
|---------------------------|------------|--------|
| **Linear Regression**      | 0.7527     | 6.3731 |
| **Random Forest**          | 0.5104     | 12.618 |
| **Support Vector Regressor** | -0.0989    | 28.324 |
| **XGBoost (Optimized)**    | 0.8763     | 5.3225 |

## Explainable AI (XAI)

### SHAP Analysis

- **SHAP Dependence Plots**: These plots show the relationship between features and the SHAP values. For example, the `Present_Price` feature has a significant positive impact on the predicted car price.
- **SHAP Force Plots**: These plots show how individual features push the predicted price higher or lower. `Present_Price` and `Driven_kms` have the highest impacts.

### Key Insights:
1. **Present_Price**: Higher present prices lead to higher predicted prices.
2. **Car_Age**: Newer cars have higher predicted prices.
3. **Driven_kms**: Fewer kilometers driven results in a higher predicted price.
4. **Fuel Type**: Diesel cars have a slightly higher price prediction compared to Petrol cars.
5. **Transmission Type**: Automatic transmission cars tend to fetch higher prices than manual transmission ones.

## Code Structure

1. **`data_preprocessing.py`**: Script for cleaning and preprocessing the data.
2. **`model_training.py`**: Script for training the machine learning models.
3. **`hyperparameter_tuning.py`**: Script for optimizing the hyperparameters of the models.
4. **`shap_analysis.py`**: Script for SHAP analysis and visualization of the model's interpretability.


## Next Steps
1. **Model Deployment**: Deploy the trained model using Flask to create a car price prediction web application.
2. **Model Improvements**: Explore adding more features (such as car condition) and retraining the model with a larger dataset.
3. **Business Use**: Implement the model into a car price prediction application or recommendation engine.
4. **Interpretability**: Use SHAP for better decision-making transparency in pricing and business processes.

## Requirements
The following Python libraries are required to run the project:

- `xgboost`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `shap`
- `sklearn`


## Conclusion

The XGBoost model with hyperparameter optimization and XAI techniques provides an accurate, interpretable, and reliable approach for predicting car prices. The implementation of SHAP values enables the understanding of the model’s decision-making process, ensuring transparency for stakeholders.
