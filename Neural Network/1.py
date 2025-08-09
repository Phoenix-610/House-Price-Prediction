import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import math

# Load the dataset
file_path = "housing.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Print initial number of samples
print(f"Number of samples before outlier removal: {len(df)}")

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Remove outliers using IQR method (reusing your function)
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

# Use StandardScaler for feature normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply log transformation to target variable (fixing the skewness)
# Adding a small constant to avoid log(0)
y_log = np.log1p(y)

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
X_train, X_val, X_test, y_train_log, y_val_log, y_test_log = split_data(X_scaled, y_log)

# Neural Network with dropout
class ImprovedNeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size=1, dropout_rate=0.2):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self.parameters = self._initialize_parameters()
        
    def _initialize_parameters(self):
        np.random.seed(42)
        parameters = {}
        
        # Xavier/Glorot initialization for better convergence
        # Input layer to first hidden layer
        parameters['W1'] = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / (self.input_size + self.hidden_size))
        parameters['b1'] = np.zeros((1, self.hidden_size))
        
        # Hidden layers
        for i in range(2, self.hidden_layers + 1):
            parameters[f'W{i}'] = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(2 / (self.hidden_size + self.hidden_size))
            parameters[f'b{i}'] = np.zeros((1, self.hidden_size))
        
        # Output layer
        parameters[f'W{self.hidden_layers + 1}'] = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / (self.hidden_size + self.output_size))
        parameters[f'b{self.hidden_layers + 1}'] = np.zeros((1, self.output_size))
        
        return parameters
    
    def leaky_relu(self, Z, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.maximum(alpha * Z, Z)
    
    def leaky_relu_derivative(self, Z, alpha=0.01):
        """Derivative of Leaky ReLU function"""
        dZ = np.ones_like(Z)
        dZ[Z < 0] = alpha
        return dZ
    
    def dropout(self, A, dropout_rate, is_training):
        """Apply dropout to activations"""
        if not is_training or dropout_rate == 0:
            return A
        
        keep_prob = 1 - dropout_rate
        mask = np.random.rand(*A.shape) < keep_prob
        A *= mask
        A /= keep_prob  # Scale to maintain expected value
        return A, mask
    
    def forward_propagation(self, X, is_training=True):
        """Forward propagation through the network"""
        cache = {'A0': X}
        dropout_masks = {}
        A = X
        
        # Process through hidden layers with Leaky ReLU activation
        for i in range(1, self.hidden_layers + 1):
            Z = np.dot(A, self.parameters[f'W{i}']) + self.parameters[f'b{i}']
            A = self.leaky_relu(Z)
            
            # Apply dropout during training
            if is_training and self.dropout_rate > 0:
                A, mask = self.dropout(A, self.dropout_rate, is_training)
                dropout_masks[f'D{i}'] = mask
            
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = A
        
        # Output layer (linear activation for regression)
        Z_out = np.dot(A, self.parameters[f'W{self.hidden_layers + 1}']) + self.parameters[f'b{self.hidden_layers + 1}']
        cache[f'Z{self.hidden_layers + 1}'] = Z_out
        cache['dropout_masks'] = dropout_masks
        
        return Z_out, cache
    
    def compute_cost(self, y_pred, y_true, regularization=0.01):
        """Calculate mean squared error cost with L2 regularization"""
        m = y_true.shape[0]
        mse_cost = (1/(2*m)) * np.sum(np.square(y_pred - y_true.reshape(-1, 1)))
        
        # L2 regularization
        l2_cost = 0
        for i in range(1, self.hidden_layers + 2):
            l2_cost += np.sum(np.square(self.parameters[f'W{i}']))
        
        l2_cost = (regularization / (2*m)) * l2_cost
        
        return mse_cost + l2_cost
    
    def backward_propagation(self, y_pred, y_true, cache, regularization=0.01):
        """Backward propagation to calculate gradients"""
        m = y_true.shape[0]
        gradients = {}
        dropout_masks = cache['dropout_masks']
        
        # Output layer gradient
        dZ_out = y_pred - y_true.reshape(-1, 1)
        gradients[f'dW{self.hidden_layers + 1}'] = (1/m) * np.dot(cache[f'A{self.hidden_layers}'].T, dZ_out)
        gradients[f'dW{self.hidden_layers + 1}'] += (regularization/m) * self.parameters[f'W{self.hidden_layers + 1}']  # L2 regularization term
        gradients[f'db{self.hidden_layers + 1}'] = (1/m) * np.sum(dZ_out, axis=0, keepdims=True)
        
        # Hidden layers gradient
        dA = np.dot(dZ_out, self.parameters[f'W{self.hidden_layers + 1}'].T)
        
        for i in range(self.hidden_layers, 0, -1):
            # Apply dropout mask if applicable
            if self.dropout_rate > 0 and f'D{i}' in dropout_masks:
                dA *= dropout_masks[f'D{i}']
                dA /= (1 - self.dropout_rate)
                
            dZ = dA * self.leaky_relu_derivative(cache[f'Z{i}'])
            gradients[f'dW{i}'] = (1/m) * np.dot(cache[f'A{i-1}'].T, dZ)
            gradients[f'dW{i}'] += (regularization/m) * self.parameters[f'W{i}']  # L2 regularization term
            gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 1:
                dA = np.dot(dZ, self.parameters[f'W{i}'].T)
        
        return gradients
    
    def update_parameters(self, gradients, learning_rate, momentum=0.9, velocity=None):
        """Update weights and biases using momentum"""
        if velocity is None:
            velocity = {}
            for i in range(1, self.hidden_layers + 2):
                velocity[f'W{i}'] = np.zeros_like(self.parameters[f'W{i}'])
                velocity[f'b{i}'] = np.zeros_like(self.parameters[f'b{i}'])
        
        for i in range(1, self.hidden_layers + 2):
            # Update with momentum
            velocity[f'W{i}'] = momentum * velocity[f'W{i}'] - learning_rate * gradients[f'dW{i}']
            velocity[f'b{i}'] = momentum * velocity[f'b{i}'] - learning_rate * gradients[f'db{i}']
            
            self.parameters[f'W{i}'] += velocity[f'W{i}']
            self.parameters[f'b{i}'] += velocity[f'b{i}']
        
        return velocity
    
    def train(self, X_train, y_train, X_val, y_val, learning_rate=0.001, 
              momentum=0.9, regularization=0.01, epochs=2000, batch_size=64, 
              patience=100, min_delta=1e-6, verbose=True):
        """Train the neural network"""
        m = X_train.shape[0]
        costs_train = []
        costs_val = []
        best_val_cost = float('inf')
        patience_counter = 0
        velocity = None
        best_parameters = None
        
        # Convert inputs to numpy arrays if they aren't already
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Learning rate scheduler
        initial_lr = learning_rate
        
        for epoch in range(epochs):
            # Learning rate decay
            learning_rate = initial_lr / (1 + 0.01 * epoch)
            
            # Mini-batch gradient descent
            indices = np.random.permutation(m)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_cost_train = 0
            
            # Process in mini-batches
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # Forward propagation
                y_pred_batch, cache = self.forward_propagation(X_batch, is_training=True)
                
                # Compute batch cost
                batch_cost = self.compute_cost(y_pred_batch, y_batch, regularization)
                epoch_cost_train += batch_cost * (end_idx - i) / m
                
                # Backward propagation
                gradients = self.backward_propagation(y_pred_batch, y_batch, cache, regularization)
                
                # Update parameters with momentum
                velocity = self.update_parameters(gradients, learning_rate, momentum, velocity)
            
            # Evaluate on validation set
            y_pred_val, _ = self.forward_propagation(X_val, is_training=False)
            val_cost = self.compute_cost(y_pred_val, y_val, regularization)
            
            costs_train.append(epoch_cost_train)
            costs_val.append(val_cost)
            
            # Early stopping check
            if val_cost < best_val_cost - min_delta:
                best_val_cost = val_cost
                patience_counter = 0
                # Save best parameters
                best_parameters = {k: v.copy() for k, v in self.parameters.items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                # Restore best parameters
                self.parameters = best_parameters
                break
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Cost: {epoch_cost_train:.6f}, Val Cost: {val_cost:.6f}, LR: {learning_rate:.6f}")
        
        return costs_train, costs_val
    
    def predict(self, X):
        """Make predictions with the trained model"""
        y_pred, _ = self.forward_propagation(X, is_training=False)
        return y_pred

# Evaluate the model
def evaluate_model(model, X, y_true_log, original_y, dataset_name):
    """Evaluate model performance"""
    # Get log predictions
    y_pred_log = model.predict(X).flatten()
    
    # Calculate metrics on log scale
    r2_log = r2_score(y_true_log, y_pred_log)
    mse_log = mean_squared_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mse_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    
    # Convert back to original scale
    y_pred_original = np.expm1(y_pred_log)
    
    # Calculate metrics on original scale
    r2 = r2_score(original_y, y_pred_original)
    mse = mean_squared_error(original_y, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_y, y_pred_original)
    
    print(f"\n{dataset_name} Performance:")
    print(f"On log scale - R² Score: {r2_log:.4f}, RMSE: {rmse_log:.4f}")
    print(f"On original scale - R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return r2, mse, rmse, mae, y_pred_original

# Create and train the improved model
input_size = X_train.shape[1]
hidden_layers = 3  
hidden_size = 128 

# Get original scale y values for evaluation
y_train_orig = np.expm1(y_train_log)
y_val_orig = np.expm1(y_val_log)
y_test_orig = np.expm1(y_test_log)

# Create improved model
nn_model = ImprovedNeuralNetwork(
    input_size=input_size,
    hidden_layers=hidden_layers, 
    hidden_size=hidden_size,
    dropout_rate=0.2  # Add dropout for regularization
)

# Train with improved parameters
costs_train, costs_val = nn_model.train(
    X_train, y_train_log, 
    X_val, y_val_log,
    learning_rate=0.001,  # Lower learning rate
    momentum=0.9,        # Add momentum
    regularization=0.01, # L2 regularization
    epochs=2000,
    batch_size=64,
    patience=100,        # Increased patience
    verbose=True
)

# Evaluate on all datasets
_, _, _, _, y_train_pred = evaluate_model(nn_model, X_train, y_train_log, y_train_orig, "Training")
_, _, _, _, y_val_pred = evaluate_model(nn_model, X_val, y_val_log, y_val_orig, "Validation")
_, _, _, _, y_test_pred = evaluate_model(nn_model, X_test, y_test_log, y_test_orig, "Test")

# Improved visualizations
plt.figure(figsize=(18, 12))

# Learning curve
plt.subplot(231)
plt.plot(costs_train, label='Training Cost')
plt.plot(costs_val, label='Validation Cost')
plt.title('Learning Curve', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error (Log Scale)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Training: Actual vs Predicted
plt.subplot(232)
plt.scatter(y_train_orig, y_train_pred, alpha=0.5, color='blue')
plt.plot([y_train_orig.min(), y_train_orig.max()], [y_train_orig.min(), y_train_orig.max()], 'r--')
plt.title('Training: Actual vs Predicted', fontsize=14)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
# Fix axes to be equal
plt.axis('equal')
plt.axis([0, max(y_train_orig.max(), y_train_pred.max())*1.05, 
          0, max(y_train_orig.max(), y_train_pred.max())*1.05])

# Validation: Actual vs Predicted
plt.subplot(233)
plt.scatter(y_val_orig, y_val_pred, alpha=0.5, color='green')
plt.plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--')
plt.title('Validation: Actual vs Predicted', fontsize=14)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
# Fix axes to be equal
plt.axis('equal')
plt.axis([0, max(y_val_orig.max(), y_val_pred.max())*1.05, 
          0, max(y_val_orig.max(), y_val_pred.max())*1.05])

# Test: Actual vs Predicted
plt.subplot(234)
plt.scatter(y_test_orig, y_test_pred, alpha=0.5, color='purple')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.title('Test: Actual vs Predicted', fontsize=14)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
# Fix axes to be equal
plt.axis('equal')
plt.axis([0, max(y_test_orig.max(), y_test_pred.max())*1.05, 
          0, max(y_test_orig.max(), y_test_pred.max())*1.05])

# Residual plot for test set
plt.subplot(235)
residuals = y_test_orig - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot (Test Set)', fontsize=14)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Error distribution
plt.subplot(236)
plt.hist(residuals, bins=30, color='teal', alpha=0.7)
plt.title('Error Distribution', fontsize=14)
plt.xlabel('Prediction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('neural_network_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance analysis - using weights magnitude as a proxy
def analyze_feature_importance(model, feature_names):
    """Analyze feature importance based on neural network weights"""
    # Get weights from first layer
    first_layer_weights = np.abs(model.parameters['W1'])
    
    # Calculate average importance for each feature
    importance = np.mean(first_layer_weights, axis=1)
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

# Get feature names
feature_names = X.columns.tolist()

# Calculate feature importance
feature_importance = analyze_feature_importance(nn_model, feature_names)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10], color='teal')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 10 Feature Importance', fontsize=14)
plt.gca().invert_yaxis()  # Highest importance at the top
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

