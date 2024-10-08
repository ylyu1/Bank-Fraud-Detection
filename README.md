## Project Overview
In this project, I collaborated with four data scientists to develop a transaction fraud detector, reducing operational costs and safeguarding the partner bank's reputation.
### Major Components of the Project:
- **Data source**: non-disclosure agreement(NDA) data from a partner bank
- **Data size**: 1.5 million transactions
- **Data Preprocessing and Exploratory Data Analysis(EDA)**: Transformed data types, investigated and disputed missing, analyzed geo location, time-series features and individual customer behavior, checked outliers. 
- **Modeling**: Addressed imbalanced data using techniques such as SMOTE and undersampling. Developed Python modules for feature generation and engineering, focusing on transaction characteristics such as velocity and frequency, and implemented feature crossing to enhance model interpretability. Compared and optimized various binary classification models including Logistic Regression, Random Forest, and XGBoost. Logistic Regression was selected as the best-performing model based on its ROC-AUC of 0.89 and good interpretability.
- **Deployment**: Deployed the model on AWS for internal use, enhancing the partner bank's ability to predict and manage fraud risk
### Dashboard
- Creating visualizations from part of the transaction data
- Using SQL server for data preprocessing
- Using Power BI in connection with SQL server for data manipulation, data visualization and dashboarding
![dashboard](dashboard.png)
