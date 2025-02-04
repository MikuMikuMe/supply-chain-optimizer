Creating a supply chain optimizer involves several steps, including data preprocessing, demand prediction using machine learning, and inventory optimization. Below is a simplified version of such a program with key components. This program uses a sample dataset, so you'll need to replace it with your actual data. The process mainly includes loading the data, preprocessing, training a demand prediction model, and optimizing inventory levels.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import sys

# Sample function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: No data found.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: File parsing failed.")
        sys.exit(1)

# Basic data preprocessing function
def preprocess_data(data):
    try:
        # Drop missing values - simple cleanup
        data = data.dropna()
        print("Data preprocessing completed.")
        return data
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        sys.exit(1)

# Function to train a demand prediction model
def train_demand_model(features, target):
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Model trained. Mean Absolute Error: {mae}")
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

# Function to optimize inventory levels
def optimize_inventory(demand_predictions, inventory_data):
    try:
        # Simple logic to set inventory - inventory should be demand * (1 + safety stock percentage)
        safety_stock_percentage = 0.15
        inventory_data['OptimalInventory'] = demand_predictions * (1 + safety_stock_percentage)
        print("Inventory optimization completed.")
        return inventory_data
    except Exception as e:
        print(f"Error during inventory optimization: {e}")
        sys.exit(1)

def main():
    # Load data
    data_path = 'supply_chain_data.csv'  # Replace with your actual CSV file path
    data = load_data(data_path)
    
    # Preprocess the data
    data = preprocess_data(data)

    # Assume 'Feature1', 'Feature2', ... are feature columns and 'Demand' is the target column
    features = data[['Feature1', 'Feature2', 'Feature3']]  # Update with actual feature columns
    target = data['Demand']  # Update with actual target column

    # Train the demand prediction model
    model = train_demand_model(features, target)

    # Get predictions
    demand_predictions = model.predict(features)

    # Optimize inventory
    inventory_data = data.copy()
    optimized_inventory = optimize_inventory(demand_predictions, inventory_data)

    # Output optimized inventory
    print(optimized_inventory.head())

if __name__ == "__main__":
    main()
```

### Key Components Explained:

1. **Data Loading**:
   - `load_data()`: Handles reading the CSV file and incorporates error handling for common file reading issues.

2. **Data Preprocessing**:
   - `preprocess_data()`: Cleans the dataset by dropping NaN values and catching potential errors during preprocessing.

3. **Demand Prediction**:
   - `train_demand_model()`: Uses a machine learning model (RandomForestRegressor) to predict demand based on provided features.

4. **Inventory Optimization**:
   - `optimize_inventory()`: Uses a basic strategy to calculate optimal inventory levels, incorporating a safety stock percentage.

5. **Error Handling**:
   - Each function tries to catch and print out any errors, exiting the program gracefully if anything goes wrong.

In a real-world application, you would refine each part, include domain-specific transformations, feature engineering, advanced model selection, and more sophisticated inventory optimization techniques (like linear programming or stochastic models). Additionally, dataset sizes, specific column names, more advanced error handling, and logging should be considered.