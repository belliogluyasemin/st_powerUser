
## Power User Modeling for Mailing Campaign

### About Data
Google Merchandise Store](https://shop.merch.google/) is an online store that sells Google-branded merchandise. The site uses Google Analytics 4's standard web ecommerce implementation along with enhanced measurement. The ga4_obfuscated_sample_ecommerce dataset available through the [BigQuery Public Datasets](https://console.cloud.google.com/bigquery) program contains a sample of obfuscated BigQuery event export data for three months from 2020-11-01 to 2021-01-31.

Welcome to the Power User Modeling for Mailing Campaign repository! This project features a Streamlit interface to analyze user behavior and predict user spending. Below you will find information about the project, data, models, and how to run the Streamlit application.

### Streamlit Interface

You can access the Streamlit interface for this project [here](https://stpoweruserapp.streamlit.app/).

Through this interface, you can analyze the user's 90-day revenue based on their first 15-day behavior and predict whether the user is likely to spend more money.

### About the Data

The dataset used in this project includes user transaction data and various features related to user behavior. 

- **LTV Prediction**: Predicting the Lifetime Value (LTV) of users over 90 days.
- **Power User Prediction**: Initially, power users were identified based on a z-score method. However, this method resulted in very few power users, making it insufficient. Therefore, based on the distributions, users who spent more than $110 were identified as power users, forming the basis for the classification target.



### LTV Prediction

We used Random Forest Regressor for predicting LTV. After hyperparameter optimization, the R2 score increased to 83%.

### Power User Prediction

### 1. Model Training and Oversampling
- **Training Dataset Preparation**: The training dataset was adjusted using various oversampling methods to handle class imbalances.
- **Oversampling Techniques**: Methods like RandomOverSample, SMOTE, and ADASYN were employed to balance the dataset, ensuring that the models could effectively learn from both the majority and minority classes.

### 2. Model Selection
- **Hyperparameter Optimization**: GridSearchCV was used for hyperparameter tuning to find the best model configurations.
- **Comparison of Models**: Several models were compared based on their performance metrics, including KNN, XGBoost, Logistic Regression, and Random Forest. The XGBoost model with ADASYN oversampling showed the best performance.


| Model                                       | Recall  | Precision | Log Loss |
|---------------------------------------------|---------|-----------|----------|
| KNN                                         | 51.65%  | 88.70%    | 6.88%    |
| KNN_RandomOverSample                        | 47.25%  | 23.50%    | 15.45%   |
| KNN_SMOTE                                   | 62.64%  | 19.40%    | 19.13%   |
| KNN_ADASYN                                  | 28.57%  | 35.60%    | 24.31%   |
| XGBoost                                     | 71.43%  | 100.00%   | 5.23%    |
| XGBoost_RandomOverSample                    | 71.43%  | 55.60%    | 2.70%    |
| XGBoost_SMOTE                               | 71.43%  | 97.00%    | 1.41%    |
| **XGBoost_ADASYN**                          | 71.43%  | 100.00%   | 1.33%    |
| Logistic Regression                         | 67.03%  | 89.70%    | 1.21%    |
| Logistic Regression_RandomOverSample        | 78.02%  | 10.50%    | 14.46%   |
| Logistic Regression_SMOTE                   | 76.92%  | 11.60%    | 12.64%   |
| Logistic Regression_ADASYN                  | 84.62%  | 6.40%     | 24.08%   |
| RandomForest                                | 70.33%  | 100.00%   | 4.49%    |
| RandomForest_RandomOverSample               | 63.74%  | 87.90%    | 5.12%    |
| RandomForest_SMOTE                          | 72.53%  | 42.00%    | 3.16%    |
| RandomForest_ADASYN                         | 74.73%  | 32.70%    | 4.43%    |


### 4. Threshold Adjustment and Model Evaluation
- **Threshold Tuning**: The threshold value of the model was adjusted to optimize performance. Despite testing various thresholds, the default value of 0.5 was retained as it provided the best balance between recall and precision.
- **ROC AUC Curve Analysis**: The performance of the XGBoost ADASYN model was further validated using the ROC AUC curve, demonstrating strong predictive capabilities.


### Running the Streamlit Application

Run the Streamlit app by uploading the `powerUser_streamlit_v002_df_pickle.py` file to Streamlit.io. The `requirements.txt` file in the repository contains all the necessary dependencies.



