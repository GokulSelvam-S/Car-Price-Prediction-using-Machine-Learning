import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("cardekho.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

df.dropna(inplace=True)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')

df.dropna(inplace=True)

df['brand'] = df['name'].str.split().str[0]

plt.figure(figsize=(8,5))
sns.histplot(df['selling_price'], bins=50, kde=True)
plt.title("Selling Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='year', y='selling_price', data=df)
plt.title("Year vs Selling Price")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='km_driven', y='selling_price', data=df)
plt.title("KM Driven vs Selling Price")
plt.show()

fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()
owner_encoder = LabelEncoder()
brand_encoder = LabelEncoder()

df['fuel'] = fuel_encoder.fit_transform(df['fuel'])
df['seller_type'] = seller_encoder.fit_transform(df['seller_type'])
df['transmission'] = trans_encoder.fit_transform(df['transmission'])
df['owner'] = owner_encoder.fit_transform(df['owner'])
df['brand'] = brand_encoder.fit_transform(df['brand'])

X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'brand']]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nLinear Regression Performance")
print("R2 Score:", round(r2_score(y_test, lr_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, lr_pred), 2))

print("\nRandom Forest Performance")
print("R2 Score:", round(r2_score(y_test, rf_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, rf_pred), 2))

joblib.dump(rf_model, "car_price_model.pkl")

joblib.dump(fuel_encoder, "fuel_encoder.pkl")
joblib.dump(seller_encoder, "seller_encoder.pkl")
joblib.dump(trans_encoder, "trans_encoder.pkl")
joblib.dump(owner_encoder, "owner_encoder.pkl")
joblib.dump(brand_encoder, "brand_encoder.pkl")

print("\nModel saved successfully!")

def predict_price(year, kms, fuel, seller, transmission, owner, brand):

    try:

        fuel_val = fuel_encoder.transform([fuel])[0]
        seller_val = seller_encoder.transform([seller])[0]
        trans_val = trans_encoder.transform([transmission])[0]
        owner_val = owner_encoder.transform([owner])[0]
        brand_val = brand_encoder.transform([brand])[0]

        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [kms],
            'fuel': [fuel_val],
            'seller_type': [seller_val],
            'transmission': [trans_val],
            'owner': [owner_val],
            'brand': [brand_val]
        })

        prediction = rf_model.predict(input_data)

        return int(prediction[0])

    except Exception as e:
        return f"Error: {e}"

print("\nSample Predictions:")

print("Predicted Price:",
      predict_price(2018, 25000, 'Petrol', 'Individual', 'Manual', 'First Owner', 'Maruti'))

print("Predicted Price:",
      predict_price(2015, 40000, 'Diesel', 'Dealer', 'Manual', 'Second Owner', 'Hyundai'))
