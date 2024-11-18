import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = {
    'Year': [2016, 2017, 2018, 2016, 2017, 2018],
    'Area': ['Region A', 'Region A', 'Region A', 'Region B', 'Region B', 'Region B'],
    'Rainfall': [500, 480, 550, 600, 580, 570],
    'Pesticides': [100, 110, 90, 120, 115, 105],
    'Temperature': [25, 24, 26, 24, 23, 25],
    'Yield': [2000, 2050, 1980, 2200, 2250, 2150]
}

df = pd.DataFrame(data)

# One-hot encode the 'Area' column
df = pd.get_dummies(df, columns=['Area'], drop_first=True)

# Specify the columns to use
columns_to_use = ['Year', 'Rainfall', 'Pesticides', 'Temperature', 'Area_Region B']
columns_y = ['Yield']

# Check for missing columns
missing_columns = [col for col in columns_to_use + columns_y if col not in df.columns]

# Add missing columns to the DataFrame
for col in missing_columns:
    # Assuming you want to add missing columns with default values (e.g., NaN)
    df.insert(loc=len(df.columns), column=col, value=pd.NA)

if all(col in df.columns for col in columns_to_use + columns_y):
    X = df[columns_to_use]
    y = df[columns_y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    dt_predictions = dt_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    dt_mse = mean_squared_error(y_test, dt_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)

    print('Decision Tree MSE:', dt_mse)
    print('Random Forest MSE:', rf_mse)
else:
    print("Columns not found in the DataFrame.")
