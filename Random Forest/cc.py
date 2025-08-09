import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# For visualization - explicitly use Agg backend which doesn't require Tkinter
import matplotlib
matplotlib.use('Agg')  # IMPORTANT: Set the backend to Agg which doesn't require GUI
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('housing.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = ['ocean_proximity']  # This is the categorical column we know about

# Handle missing values separately for numerical and categorical columns
# For numerical columns: use mean imputation
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# For categorical column: use most frequent value
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
# Create new features that might improve model performance
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Replace infinities that might have been created during division
df = df.replace([np.inf, -np.inf], np.nan)
# Fill these new NaN values
for col in df.columns:
    if col in numerical_cols or col in ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

# Define features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Identify categorical features and numerical features
categorical_features = [col for col in X.columns if col in categorical_cols]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create column transformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest Regression model
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    ))
])

# Train the model
print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Results:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Create visualizations

# 1. Actual vs Predicted Values Plot
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.6, c='blue', edgecolor='k', linewidth=0.5)

# Add a perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

plt.title('Random Forest: Actual vs Predicted Housing Values', fontsize=16)
plt.xlabel('Actual House Value ($)', fontsize=14)
plt.ylabel('Predicted House Value ($)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add text with model performance metrics
plt.figtext(0.15, 0.8, f"RMSE: ${rmse:.2f}\nR²: {r2:.4f}\nMAE: ${mae:.2f}", fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('random_forest_predictions.png', dpi=300, bbox_inches='tight')
print("Actual vs Predicted plot saved as 'random_forest_predictions.png'")

# 2. Feature Importance Plot
# Extract feature importances
final_rf_model = rf_model.named_steps['regressor']
feature_names = []

# Get one-hot encoded categorical feature names
if len(categorical_features) > 0:
    ohe = rf_model.named_steps['preprocessor'].transformers_[0][1].named_steps['onehotencoder']
    cat_feature_names = []
    for i, cat in enumerate(categorical_features):
        encoded_features = ohe.get_feature_names_out([cat])
        cat_feature_names.extend(encoded_features)
    feature_names.extend(cat_feature_names)

# Add numerical feature names
feature_names.extend(numerical_features)

# Plot feature importances
plt.figure(figsize=(12, 10))
importances = final_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Make sure we have the correct number of feature names
if len(feature_names) != len(importances):
    # Use indices as feature names if there's a mismatch
    feature_names = [f"Feature {i}" for i in range(len(importances))]

plt.title('Random Forest: Feature Importances', fontsize=16)
plt.bar(range(len(importances)), importances[indices], align='center', color='green', alpha=0.7)
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, min(10, len(importances))])  # Show top 10 features or all if less than 10
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'random_forest_feature_importance.png'")

# 3. Residual Plot (prediction errors)
plt.figure(figsize=(12, 8))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, c='red')
plt.axhline(y=0, color='black', linestyle='-')
plt.title('Random Forest: Residual Plot', fontsize=16)
plt.xlabel('Predicted Values ($)', fontsize=14)
plt.ylabel('Residuals (Actual - Predicted) ($)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('random_forest_residuals.png', dpi=300, bbox_inches='tight')
print("Residual plot saved as 'random_forest_residuals.png'")

# Save predictions to CSV for external use
prediction_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Difference': abs(y_test - y_pred)
})
prediction_df.to_csv('rf_predictions.csv', index=False)
print("Saved predictions to 'rf_predictions.csv'")

# Example of using the model for prediction
print("\nExample predictions for the first 5 test samples:")
example_predictions = rf_model.predict(X_test.iloc[:5])
actual_values = y_test.iloc[:5].values
for i, (pred, actual) in enumerate(zip(example_predictions, actual_values)):
    print(f"Sample {i+1}: Predicted: ${pred:.2f}, Actual: ${actual:.2f}, Difference: ${abs(pred-actual):.2f}")