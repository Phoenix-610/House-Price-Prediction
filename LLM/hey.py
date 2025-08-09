

import pandas as pd  
import google.generativeai as genai  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
import numpy as np  
import re  
print("Starting script...")  
# Your code  

# ==== STEP 1: Set up Gemini API ====  
genai.configure(api_key="AIzaSyBdUB4PFEDHYIUkG_z2JSpKyjEDzfubl30")  # Replace with your actual API key  

# Use the correct model name format  
model = genai.GenerativeModel("models/gemini-1.5-pro")  # Replace with the correct identifier for the model  

# ==== STEP 2: Load and preprocess dataset ====  
df = pd.read_csv("Housing.csv")  # Ensure this file exists in your directory  

# Fill missing values  
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())  

# Encode categorical features  
le = LabelEncoder()  
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])  

# Split into features (X) and target (y)  
X = df.drop(columns=['median_house_value'])  
y = df['median_house_value']  

# Train-test split (80% train, 20% test)  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  

# Combine into dataframes  
train_df = X_train.copy()  
train_df['median_house_value'] = y_train  

test_df = X_test.copy()  
test_df['median_house_value'] = y_test  

# ==== STEP 3: Create few-shot prompt ====  
few_shot_train = train_df.sample(100, random_state=42)  
few_shot_test = test_df.sample(5, random_state=42)  
true_values = few_shot_test['median_house_value'].values  
few_shot_test_features = few_shot_test.drop(columns=['median_house_value'])  

# Format prompt  
prompt = "You are a helpful AI that predicts house prices based on input features.\n\n"  
prompt += "Here are some examples:\n"  
for row in few_shot_train.to_dict(orient="records"):  
    input_features = {k: v for k, v in row.items() if k != 'median_house_value'}  
    prompt += f"Input: {input_features}\n"  
    prompt += f"Output: {row['median_house_value']}\n\n"  

prompt += "Now predict prices for the following houses:\n"  
for index, row in few_shot_test_features.iterrows():  
    prompt += f"Input: {row.to_dict()}\n"  

# ==== STEP 4: Get Gemini prediction ====  
response = model.generate_content(prompt)  
response_text = response.text  
print("🔮 Gemini Response:\n", response_text)  

# Save the response to a text file  
with open("gemini_response.txt", "w") as file:  
    file.write(response_text)  

# ==== STEP 5: Extract predicted prices ====  
# After extracting predicted prices:  
predicted_prices = []  

# Regex for extracting price ranges  
price_ranges = re.findall(r'Around \$([\d,]+) - \$([\d,]+)', response_text)  

# Loop through the found price ranges to extract values  
for index, (lower_bound, upper_bound) in enumerate(price_ranges):  
    # Convert string prices to float, removing any commas  
    lower_price = float(lower_bound.replace(',', ''))  
    upper_price = float(upper_bound.replace(',', ''))  
    
    # Calculate the average price  
    average_price = (lower_price + upper_price) / 2  
    predicted_prices.append(average_price)  

    # Print the prediction along with the row number  
    print(f"Predicted Price for Row {index + 1}: ${average_price:.2f}")  

print("Extracted Predicted Prices:", predicted_prices)   

# Check if we have predictions  
if len(predicted_prices) == 0:  
    print("No predicted prices were extracted. Please check the model response format.")  
else:  
    # Ensure we continue with evaluation and plotting only if prices are present  
    if len(true_values) != len(predicted_prices):  
        print("The number of true values does not match the number of predicted prices.")  
    else:  
        # Evaluate  
        mae = mean_absolute_error(true_values, predicted_prices)  
        rmse = np.sqrt(mean_squared_error(true_values, predicted_prices))  
        r2 = r2_score(true_values, predicted_prices)  
        
        print("\n📊 Evaluation Metrics:")  
        print(f"MAE: {mae:.2f}")  
        print(f"RMSE: {rmse:.2f}")  
        print(f"R² Score: {r2:.4f}")  

        # Print the summary table of true values and predicted prices  
        print("\nSummary of Predictions:")  
        print(f"{'Row':<5} {'True Value':<15} {'Predicted Price':<15}")  
        for idx in range(len(true_values)):  
            print(f"{idx + 1:<5} ${true_values[idx]:<14.2f} ${predicted_prices[idx]:<14.2f}")
            

            
 