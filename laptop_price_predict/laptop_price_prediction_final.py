import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset
laptop_data = pd.read_csv('C:\\Users\\YOGINATH\\Desktop\\laptop_price_predict\\laptop.csv')
laptop_data.head()

# Data info and description
laptop_data.info()
laptop_data.describe()

# Droping rows with missing values
laptop_data = laptop_data.dropna()

# Droping unnecessary columns
laptop_data = laptop_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Weight', 'ScreenResolution', 'TypeName'], errors='ignore')

# Converting 'Ram' column to integer
laptop_data['Ram'] = laptop_data['Ram'].str.replace('GB', '').astype(int)

# Applying log transformation to positively skewed columns
laptop_data['Price'] = np.log1p(laptop_data['Price'])

# Converting categorical columns to dummy variables
categorical_cols = ['Company', 'OpSys', 'Cpu', 'Gpu', 'Memory']
laptop_data = pd.get_dummies(laptop_data, columns=categorical_cols, drop_first=True)

# Replacing '?' with NaN and drop rows with missing values
laptop_data = laptop_data.replace('?', np.nan)
laptop_data = laptop_data.dropna()

# Splitting data into features (X) and target (y)
X = laptop_data.drop(columns=['Price'])
y = laptop_data['Price']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Checking the type of X_train
print(type(X_train))  # This should print: <class 'pandas.core.frame.DataFrame'>

# Initializing and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')



# Function to predict price for new laptop input
def predict_price(input_data, model, X_train):
    # Ensure X_train is a DataFrame
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a DataFrame, not a list.")
    
    # Creating a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensuring the input data has the same columns as the training data
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # Predicting the price using the trained model
    prediction = model.predict(input_df)
    return prediction[0]

# Example input for a new laptop
new_laptop = {
    'Inches': 15.6,
    'Ram': 4,
    '500GB HDD': 1,
    'Company_HP': 1,
    'Company_Chuwi': 0,
    'OpSys_Windows': 1,
    'OpSys_macOS': 0,
    'Processor_i3': 1,  # Example encoding for processor
    'Intel HD Graphics 520': 1     # Example encoding for GPU
}

# Making prediction for the new laptop
predicted_price_log = predict_price(new_laptop, model, X_train)
predicted_price = np.exp(predicted_price_log)  # Reverse log transformation
print(f'Predicted Price for the new laptop: ${predicted_price:.2f}')
