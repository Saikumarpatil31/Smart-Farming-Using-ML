from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
# df = pd.read_csv('your_dataset.csv')

data = {
    'Year': [2016, 2017, 2018, 2019, 2020, 2021],
    'Area': ['Region A', 'Region B', 'Region C', 'Region D', 'Region E', 'Region F'],
    'Rainfall': [500, 480, 550, 600, 580, 570],
    'Pesticides': [100, 110, 90, 120, 115, 105],
    'Temperature': [25, 24, 26, 24, 23, 25],
    'Yield': [2000, 2050, 1980, 2200, 2250, 2150]
}

df = pd.DataFrame(data)

# One-hot encode the 'Area' column
df = pd.get_dummies(df, columns=['Area'], drop_first=True)

# Assuming you have already preprocessed your data
# Replace the below line with your actual data preprocessing code if needed
columns_to_use = ['Year', 'Rainfall', 'Pesticides', 'Temperature']
X = df[columns_to_use]
y = df['Yield']  # Separate target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    year = int(request.form['year'])
    rainfall = float(request.form['rainfall'])
    pesticides = float(request.form['pesticides'])
    temperature = float(request.form['temperature'])

    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'Year': [year],
        'Rainfall': [rainfall],
        'Pesticides': [pesticides],
        'Temperature': [temperature]
    })

    # Use the trained model to make a prediction
    prediction = model.predict(input_data)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
