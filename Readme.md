# 🏡 Multi-Model House Price Prediction 📈

A comprehensive data science project that predicts housing prices using three distinct models: a **Random Forest Regressor**, a **custom-built Neural Network**, and Google's **Gemini LLM**. This repository provides a comparative analysis of traditional machine learning, deep learning, and large language model approaches for a regression task.

![divider](https://i.imgur.com/3iEaB8G.png)

## ✨ Features

- **Multi-Model Implementation**: Compares three powerful and distinct models.
- **In-Depth Analysis**: Includes feature engineering, outlier removal, and detailed performance evaluation.
- **Custom Neural Network**: A NN built from scratch using NumPy to showcase foundational deep learning concepts.
- **LLM for Regression**: Explores the use of the Gemini Pro model for price prediction via few-shot prompting.
- **Rich Visualizations**: Generates plots for actual vs. predicted values, feature importance, and error analysis.

![divider](https://i.imgur.com/3iEaB8G.png)

## 🤖 Models Overview

This project tackles the prediction task from three different angles:

### 1. 🌳 Random Forest Regressor
A robust ensemble learning model from `scikit-learn`. It's great for handling complex datasets and provides insights into feature importance.
- **File**: `cc.py`
- **Key Steps**:
    - Handles missing values using mean imputation.
    - Creates new features like `rooms_per_household` and `population_per_household`.
    - Uses a `Pipeline` to combine one-hot encoding for categorical features with the regressor.
    - Evaluates the model using MAE, RMSE, and R² score.
    - Generates and saves plots for feature importance and prediction residuals.

### 2. 🧠 Custom Neural Network
A deep learning model built from scratch to demonstrate the inner workings of a neural network.
- **File**: `1.py`
- **Key Steps**:
    - Implements advanced techniques like Leaky ReLU, Dropout, L2 Regularization, and Momentum.
    - Uses a learning rate scheduler and early stopping for efficient training.
    - Normalizes features using `StandardScaler` and applies a log transform to the target variable to handle skewness.
    - Provides a comprehensive evaluation on training, validation, and test sets.

### 3. 🔮 Gemini Large Language Model (LLM)
An innovative approach that uses Google's `gemini-1.5-pro` model to predict prices based on a few-shot prompt.
- **File**: `hey.py`
- **Key Steps**:
    - Constructs a detailed prompt with 100 examples from the training set.
    - Asks the model to predict prices for 5 new examples from the test set.
    - Parses the natural language response from the LLM to extract numerical predictions.
    - Evaluates the LLM's performance against the true values.

![divider](https://i.imgur.com/3iEaB8G.png)

## 📊 Dataset

This project uses the famous **California Housing Prices** dataset. It contains information from the 1990 California census. The goal is to predict the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

You can typically find this dataset on Kaggle or download it directly. Ensure you have `housing.csv` in the root directory of the project.

![divider](https://i.imgur.com/3iEaB8G.png)

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python 3.x installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn matplotlib google-generativeai
    ```

4.  **Set up your Gemini API Key:**
    For the `hey.py` script, you need to configure your Gemini API key. Find the following line and replace the placeholder with your actual key:
    ```python
    # In hey.py
    genai.configure(api_key="YOUR_API_KEY")
    ```

![divider](https://i.imgur.com/3iEaB8G.png)

## 🛠️ Usage

Make sure the `housing.csv` dataset is in the project's root folder. Then, you can run each model's script from your terminal.

-   **To run the Random Forest model:**
    ```sh
    python cc.py
    ```
    This will train the model, print evaluation metrics, and save three PNG images: `random_forest_predictions.png`, `random_forest_feature_importance.png`, and `random_forest_residuals.png`.

-   **To run the Custom Neural Network model:**
    ```sh
    python 1.py
    ```
    This will train the network from scratch, print detailed performance metrics for each data split, and save two PNG images: `neural_network_results.png` and `feature_importance.png`.

-   **To run the Gemini LLM model:**
    ```sh
    python hey.py
    ```
    This will send a request to the Gemini API, print the model's textual response, and then show the extracted numerical predictions and evaluation metrics.

![divider](https://i.imgur.com/3iEaB8G.png)

## 📈 Results

The scripts automatically calculate and display performance metrics for each model. Here's an example of the kind of visualizations this project generates:

| Actual vs. Predicted (Random Forest) | Feature Importance (Neural Network) |
| :----------------------------------: | :-----------------------------------: |
| ![Actual vs Predicted Plot](https://i.imgur.com/your-placeholder-1.png) | ![Feature Importance Plot](https://i.imgur.com/your-placeholder-2.png) |

*(Note: Replace the placeholder image URLs above with screenshots of the plots generated by your scripts for a more impressive README!)*

A comparative analysis of the R² scores and RMSE from each model will show the trade-offs between interpretability (Random Forest), control (Custom NN), and the zero-shot/few-shot power of modern LLMs.

![divider](https://i.imgur.com/3iEaB8G.png)

## 📂 File Structure


.
├── 1.py                      # Script for the Custom Neural Network
├── cc.py                     # Script for the Random Forest model
├── hey.py                    # Script for the Gemini LLM model
├── housing.csv               # The dataset file
├── README.md                 # You are here!
└── requirements.txt          # (Optional) A file listing dependencies


![divider](https://i.imgur.com/3iEaB8G.png)

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

![divider](https://i.imgur.com/3iEaB8G.png)

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

(You'll need to create a file named `LICENSE` and add the MIT License text to it).
