import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "housing.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Print initial number of samples
print(f"Number of samples before outlier removal: {len(df)}")

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Remove outliers using IQR method
def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    cleaned_df = df.copy()
    
    print("\nOutlier removal details:")
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        
        if len(outliers) > 0:
            print(f"\n{column}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Outlier boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & 
                               (cleaned_df[column] <= upper_bound)]
    
    return cleaned_df

# Remove outliers
filtered_df = remove_outliers(df)

# Print number of samples after outlier removal
print(f"Number of samples after outlier removal: {len(filtered_df)}")
print(f"Number of outliers removed: {len(df) - len(filtered_df)}")
print(f"Percentage of data retained: {(len(filtered_df)/len(df))*100:.2f}%")

# Extract feature matrix (X) and target vector (y)
X = filtered_df.drop(columns=['median_house_value'])
y = filtered_df['median_house_value'].values

# Print dataset information
print(f"Total number of samples: {len(X)}")

# Use StandardScaler for normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check feature values after scaling
print("Feature sample after scaling:", X_scaled[:5])

# Comprehensive data splitting
def split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio/(1-test_ratio), random_state=42
    )
    
    print("\nDataset Split:")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.2f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.2f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.2f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Split the data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Gradient Descent with early stopping
def gradient_descent(X, y, theta, learning_rate, num_iterations, tolerance=1e-6):
    m = len(y)
    cost_history = []
    
    for i in range(num_iterations):
        predictions = X @ theta
        gradient = (1 / m) * (X.T @ (predictions - y))
        theta -= learning_rate * gradient
        
        cost = np.mean((predictions - y) ** 2)
        cost_history.append(cost)

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.2f}")
        
        # Stricter early stopping
        if i > 1 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Early stopping at iteration {i}")
            break
    
    return theta, cost_history

# Hyperparameters
learning_rate = 0.5  # Increased for faster convergence
num_iterations = 5000  # Reduced for efficiency

# Initialize theta
theta = np.zeros(X_train.shape[1])

# Train the model
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Evaluation function
def evaluate_model(X, y, theta, dataset_name):
    y_pred = X @ theta
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n{dataset_name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return r2, mse, rmse, mae

# Evaluate on training, validation, and test sets
evaluate_model(X_train, y_train, theta, "Training")
evaluate_model(X_val, y_val, theta, "Validation")
evaluate_model(X_test, y_test, theta, "Test")

# Visualization
plt.figure(figsize=(15, 5))

# Cost Reduction
plt.subplot(131)
plt.plot(cost_history)
plt.title('Cost Reduction')
plt.xlabel('Iterations')
plt.ylabel('Cost')

# Training: Actual vs Predicted
plt.subplot(132)
y_train_pred = X_train @ theta
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title('Training: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Test: Actual vs Predicted
plt.subplot(133)
y_test_pred = X_test @ theta
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Test: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()