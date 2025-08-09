# 🏡 Multi-Model House Price Prediction 📈

A comprehensive data science project that predicts housing prices using four distinct models: a **Linear Regression from scratch**, a **Random Forest Regressor**, a **custom-built Neural Network**, and Google's **Gemini LLM**. This repository provides a comparative analysis of foundational, traditional, deep learning, and large language model approaches for a regression task.


## ✨ Features

- **Multi-Model Implementation**: Compares four powerful and distinct models, from a simple linear regressor to a complex LLM.
- **In-Depth Analysis**: Includes feature engineering, outlier removal, and detailed performance evaluation across all models.
- **From-Scratch Models**: Features a Linear Regression and a Neural Network built using only NumPy to showcase foundational concepts.
- **LLM for Regression**: Explores the use of the Gemini Pro model for price prediction via few-shot prompting.
- **Rich Visualizations**: Generates plots for actual vs. predicted values, feature importance, and error analysis.



## 🤖 Models Overview

This project tackles the prediction task from four different angles:

### 1. 📉 Linear Regression (from Scratch)
A foundational model built with NumPy to establish a baseline. This implementation uses gradient descent to find the optimal parameters for prediction.
- **File**: `4.py`
- **Key Steps**:
    - Implements outlier removal using the IQR method.
    - Normalizes features using `StandardScaler`.
    - Builds a gradient descent algorithm from scratch with early stopping to optimize the model.
    - Evaluates performance on training, validation, and test sets.

### 2. 🌳 Random Forest Regressor
A robust ensemble learning model from `scikit-learn`. It's great for handling complex datasets and provides insights into feature importance.
- **File**: `cc.py`
- **Key Steps**:
    - Handles missing values using mean imputation.
    - Creates new features like `rooms_per_household` and `population_per_household`.
    - Uses a `Pipeline` to combine one-hot encoding for categorical features with the regressor.
    - Evaluates the model using MAE, RMSE, and R² score.
    - Generates and saves plots for feature importance and prediction residuals.

### 3. 🧠 Custom Neural Network
A deep learning model built from scratch to demonstrate the inner workings of a neural network.
- **File**: `1.py`
- **Key Steps**:
    - Implements advanced techniques like Leaky ReLU, Dropout, L2 Regularization, and Momentum.
    - Uses a learning rate scheduler and early stopping for efficient training.
    - Normalizes features using `StandardScaler` and applies a log transform to the target variable to handle skewness.
    - Provides a comprehensive evaluation on training, validation, and test sets.

### 4. 🔮 Gemini Large Language Model (LLM)
An innovative approach that uses Google's `gemini-1.5-pro` model to predict prices based on a few-shot prompt.
- **File**: `hey.py`
- **Key Steps**:
    - Constructs a detailed prompt with 100 examples from the training set.
    - Asks the model to predict prices for 5 new examples from the test set.
    - Parses the natural language response from the LLM to extract numerical predictions.
    - Evaluates the LLM's performance against the true values.



## 📊 Dataset

This project uses the famous **California Housing Prices** dataset. It contains information from the 1990 California census. The goal is to predict the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

You can typically find this dataset on Kaggle or download it directly. Ensure you have `housing.csv` in the root directory of the project.


## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python 3.x installed on your system.


## 🛠️ Usage

Make sure the `housing.csv` dataset is in the project's root folder. Then, you can run each model's script from your terminal.

-   **To run the Linear Regression model:**
    ```sh
    python 4.py
    ```
    This will train the model from scratch and display plots for cost reduction and actual vs. predicted values.

-   **To run the Random Forest model:**
    ```sh
    python cc.py
    ```
    This will train the model, print evaluation metrics, and save three PNG images: `random_forest_predictions.png`, `random_forest_feature_importance.png`, and `random_forest_residuals.png`.

-   **To run the Custom Neural Network model:**
    ```sh
    python 1.py
    ```
    This will train the network from scratch, print detailed performance metrics, and save two PNG images: `neural_network_results.png` and `feature_importance.png`.

-   **To run the Gemini LLM model:**
    ```sh
    python hey.py
    ```
    This will send a request to the Gemini API, print the model's textual response, and then show the extracted numerical predictions and evaluation metrics.


