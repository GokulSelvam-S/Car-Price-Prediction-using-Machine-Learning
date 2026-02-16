# 🚗 Car Price Prediction using Machine Learning

## Overview
This project develops a Machine Learning model to predict the selling price of used cars based on key features such as manufacturing year, kilometers driven, fuel type, seller type, transmission, ownership history, and car brand. The system uses regression algorithms to estimate prices accurately and demonstrates a complete end-to-end Machine Learning workflow.

The project highlights practical skills in data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and building a prediction function using Python and Scikit-learn.

---

## Dataset Description
The dataset contains structured information about used cars with the following features:

- **Name** – Car name and model  
- **Year** – Manufacturing year  
- **Selling Price** – Price of the car (target variable)  
- **KM Driven** – Total kilometers driven  
- **Fuel** – Fuel type (Petrol, Diesel, CNG)  
- **Seller Type** – Individual or Dealer  
- **Transmission** – Manual or Automatic  
- **Owner** – Ownership history  

This dataset is commonly used for regression-based price prediction problems.

---

## Technologies and Libraries Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Joblib  

These tools were used for data processing, visualization, model training, and evaluation.

---

## Machine Learning Approach

The following regression models were implemented and evaluated:

- Linear Regression  
- Random Forest Regressor  

Random Forest provided superior performance due to its ability to capture non-linear relationships and feature interactions.

---

## Project Workflow

The project follows a standard Machine Learning pipeline:

- Data loading and inspection  
- Data cleaning and preprocessing  
- Feature engineering (extracting brand from car name)  
- Encoding categorical features  
- Exploratory Data Analysis using visualizations  
- Feature selection  
- Train-test split  
- Model training using regression algorithms  
- Model evaluation using R² Score and Mean Absolute Error  
- Building a prediction function for real-time price estimation  

---

## Model Performance

The models were evaluated using R² Score:

| Model | R² Score |
|------|----------|
| Linear Regression | 0.65 – 0.75 |
| Random Forest Regressor | 0.85 – 0.95 |

Random Forest achieved the best performance and was selected as the final model.

---

## Example Prediction

Example input values:

- Year: 2018  
- KM Driven: 25000  
- Fuel: Petrol  
- Seller Type: Individual  
- Transmission: Manual  
- Owner: First Owner  
- Brand: Maruti  

Output:

Predicted Selling Price based on trained Machine Learning model.

---

## How to Run the Project

Clone the repository:
git clone https://github.com/GokulSelvam-S/Car-Price-Prediction-using-Machine-Learning.git

Navigate to the project folder:
cd Car-Price-Prediction-using-Machine-Learning

Install required dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

Run the project:
python car_price_prediction.py

The script will train the model, evaluate performance, and display predicted prices.

---

## Project Structure

Car-Price-Prediction-using-Machine-Learning/
│
├── car_price_prediction.py
├── cardekho.csv
├── README.md
├── .gitignore

---

## Skills Demonstrated

- Machine Learning Model Development  
- Data Preprocessing and Cleaning  
- Feature Engineering  
- Regression Analysis  
- Model Evaluation  
- Data Visualization  
- Python Programming  
- End-to-End Machine Learning Pipeline  

---

## Project Objective

The objective of this project is to build an accurate and reliable Machine Learning model for predicting used car prices and demonstrate practical skills required for real-world Machine Learning applications.

---

## Author

**Gokul Selvam**

GitHub:  
https://github.com/GokulSelvam-S

---

## Support

If you found this project helpful, consider giving it a star on GitHub.
