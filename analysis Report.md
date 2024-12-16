# **Analysis Report: Car Price Prediction Model Using XGBoost**

## **1. Objective**
The goal of this project was to build a machine learning model capable of predicting the price of used cars based on various features. These features include the car’s year of manufacture, present price, mileage, fuel type, transmission type, and other characteristics. The model uses historical data to predict the selling price of a car.

## **2. Dataset Overview**
The dataset contains information about 299 used cars, with the following features:

- **Car_Name**: The name of the car
- **Year**: The year of manufacture
- **Selling_Price**: The price at which the car was sold (target variable)
- **Present_Price**: The current market price
- **Driven_kms**: The distance driven by the car (in kilometers)
- **Fuel_Type**: Type of fuel (Diesel, Petrol, etc.)
- **Selling_type**: Whether the selling type is 'Individual' or 'Dealer'
- **Transmission**: Whether the car has a manual or automatic transmission
- **Owner**: The number of previous owners

The dataset is processed, and categorical variables (such as `Fuel_Type`, `Transmission`, `Selling_type`) are one-hot encoded.

## **3. Data Preprocessing**
Before building the model, several preprocessing steps were performed:

- **Feature Engineering**: The `Year` feature was converted to `Car_Age` by subtracting the year of manufacture from the current year. This makes the feature more meaningful for predicting the car's price.
- **One-Hot Encoding**: Categorical features like `Fuel_Type`, `Selling_type`, and `Transmission` were one-hot encoded, transforming them into binary columns.
- **Scaling**: Continuous numerical features (such as `Driven_kms`, `Present_Price`) were scaled to ensure they are on a similar range to avoid dominance by any specific feature.

## **4. Model Selection**
Four regression models were initially chosen to evaluate their performance on the dataset:

1. **Linear Regression**: A simple regression model that assumes a linear relationship between features and the target variable.
2. **Random Forest Regressor**: An ensemble method that builds multiple decision trees and averages their outputs.
3. **Support Vector Regressor (SVR)**: A model that aims to fit the data within a certain margin and penalizes outliers.
4. **XGBoost Regressor**: An optimized gradient boosting algorithm known for its high performance on structured data.

## **5. Hyperparameter Tuning with GridSearchCV**
Hyperparameter tuning was performed using **GridSearchCV** to identify the best hyperparameters for the XGBoost model. The following parameters were optimized:

- **Learning rate**: Controls how much to change the model with each iteration.
- **Max depth**: The maximum depth of a tree; it helps control the model's complexity.
- **Number of estimators**: The number of trees in the model.

The best parameters identified were:

- **Learning rate**: 0.3
- **Max depth**: 3
- **Number of estimators**: 150

These parameters provided the best performance according to the **R² score** and **mean squared error (MSE)**.

## **6. Model Evaluation**
After training with the optimal hyperparameters, the models were evaluated on the test dataset. The evaluation metrics used were:

- **R² Score**: A measure of how well the model's predictions match the actual values. An R² score close to 1 indicates a good fit.
- **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values. A lower MSE indicates better accuracy.

The following results were obtained:

| Model                     | R² Score   | MSE    |
|---------------------------|------------|--------|
| **Linear Regression**      | 0.7527     | 6.3731 |
| **Random Forest**          | 0.5104     | 12.618 |
| **Support Vector Regressor** | -0.0989    | 28.324 |
| **XGBoost (Optimized)**    | 0.8763     | 5.3225 |

From the comparison, **XGBoost** with the optimal hyperparameters significantly outperforms the other models, achieving the highest **R² score** and the lowest **MSE**.

## **7. Model Insights**
- The **XGBoost** model has the highest predictive accuracy, indicating that it is the best suited for this problem.
- Feature importance was analyzed using SHAP (SHapley Additive exPlanations). Key features influencing the car price prediction include:
  - **Present_Price**: A higher present price leads to a higher predicted selling price.
  - **Year**: Newer cars tend to have higher selling prices.
  - **Driven_kms**: Cars with lower kilometers driven tend to fetch higher prices.
  - **Fuel Type**: Diesel and Petrol have distinct impacts on the price, with Diesel generally showing a higher influence.
  - **Transmission**: Manual transmission cars slightly reduce the price compared to automatic ones.

## **8. Explainable AI (XAI)**

### **8.1 Overview of Explainable AI (XAI)**
Explainable AI (XAI) is a crucial component in the responsible deployment of machine learning models, especially for high-stakes applications like pricing predictions. XAI seeks to make the decision-making process of complex models more transparent, interpretable, and understandable to humans. In this project, **XGBoost** is used for car price prediction, and XAI techniques are employed to explain the model's predictions.

### **8.2 SHAP Analysis**
SHAP values were used to explain the predictions made by the **XGBoost** model. SHAP (SHapley Additive exPlanations) values provide insights into how each feature contributes to a particular prediction. The two primary types of SHAP plots used are:

- **SHAP Dependence Plots**: These plots show the relationship between a feature and the SHAP values. They help in understanding how changes in a feature affect the model's prediction. For example, the dependence plot for **Present_Price** shows that higher present prices lead to higher predicted prices for the car.
  
- **SHAP Force Plots**: These provide a visualization of how different features push the model’s output either higher or lower. The force plot demonstrates how much each feature contributes positively or negatively to the prediction of a specific car's price.

#### **Key Insights from XAI:**
1. **Present_Price**: This feature has the highest positive impact on the predicted car price. Cars with higher present prices are likely to be predicted with higher selling prices, as shown by the SHAP force plot.
2. **Year**: Newer cars increase the predicted price. Older cars tend to have a lower predicted price.
3. **Driven_kms**: This feature negatively impacts the predicted price. Cars with fewer kilometers driven are predicted to have higher prices.
4. **Fuel Type and Transmission**: These categorical variables also influence the price, with Diesel fuel and automatic transmission having a higher contribution to the price compared to Petrol fuel and manual transmission.

### **8.3 Impact of XAI**
The use of SHAP for model interpretation ensures that the predictions of the XGBoost model are transparent and understandable. This is crucial for stakeholders who need to trust the model's decision-making process, such as car sellers or buyers, regulatory bodies, or data scientists working to improve the model.

## **9. Next Steps**
With the optimized XGBoost model yielding strong performance and providing interpretable insights via XAI, the next steps for the project are:

1. **Model Deployment**: The trained model can be deployed as a web service (e.g., using Flask) to allow users to input car features and receive predictions.
2. **Model Improvements**: If more data is available, further tuning of the model and adding more features (such as car condition or brand value) could further improve the prediction accuracy.
3. **Business Use**: The model can be integrated into a used car price prediction application or a recommendation engine for car buyers and sellers.
4. **Model Interpretability for Business Use**: Using SHAP for business stakeholders can provide explanations behind price predictions, helping them make informed decisions in car pricing.

