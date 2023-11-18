
#%%

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#%%

# getting all the candidate models
def get_valid_feature_combinations(data, features, target, low_limit, up_limit):
    # Combine features with target
    all_features = features + [target]
    
    # Calculate correlations between features and the target
    correlations = data[all_features].corr()[target]
    
    # Filter features based on the low_limit
    valid_features = [feat for feat in features if abs(correlations[feat]) >= low_limit]
    
    # Generate all combinations of valid features
    valid_combinations = []
    
    for r in range(1, len(valid_features) + 1):
        for combo in list(combinations(valid_features, r)):
            pass
            # Check the correlation between each pair of features in the combination
            if all(abs(data[feat1].corr(data[feat2])) <= up_limit for feat1, feat2 in combinations(combo, 2)):
                valid_combinations.append(list(combo))
    
    return valid_combinations

#%%

def train_and_evaluate_models(train_data, test_data, target, feature_combinations):
    results = pd.DataFrame(columns=['Features', 'Train_MSE', 'Train_R2_Adjusted', 'Test_MSE', 'Test_R2_Adjusted'])
    
    for features in feature_combinations:
        # Extract the features and target for training
        X_train = train_data[features]
        y_train = train_data[target]
        
        # Extract the features and target for testing
        X_test = test_data[features]
        y_test = test_data[target]
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on the training set
        train_predictions = model.predict(X_train)
        
        # Calculate MSE and R-squared adjusted for training data
        train_mse = mean_squared_error(y_train, train_predictions)
        train_r2_adjusted = 1 - (1 - r2_score(y_train, train_predictions)) * (len(y_train) - 1) / (len(y_train) - len(features) - 1)
        
        # Make predictions on the test set
        test_predictions = model.predict(X_test)
        
        # Calculate MSE and R-squared adjusted for testing data
        test_mse = mean_squared_error(y_test, test_predictions)
        test_r2_adjusted = 1 - (1 - r2_score(y_test, test_predictions)) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
        
        # Append results to the DataFrame
        results = results.append({
            'Features': features,
            'Train_MSE': train_mse,
            'Train_R2_Adjusted': train_r2_adjusted,
            'Test_MSE': test_mse,
            'Test_R2_Adjusted': test_r2_adjusted
        }, ignore_index=True)
    
    return results

#%%

def plot_mse_bar_chart(model_results):
    # Sort the DataFrame based on Test MSE
    sorted_results = model_results.sort_values(by='Test_MSE')
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    # Set the positions of bars on X-axis
    r1 = range(len(sorted_results))
    r2 = [x + bar_width for x in r1]
    
    # Plotting the bars
    plt.bar(r1, sorted_results['Train_MSE'], width=bar_width, edgecolor='grey', label='Train MSE')
    plt.bar(r2, sorted_results['Test_MSE'], width=bar_width, edgecolor='grey', label='Test MSE')
    
    # Adding labels
    plt.xlabel('Feature Combinations', fontweight='bold')
    plt.ylabel('MSE', fontweight='bold')
    
    # Set X-axis ticks and labels
    plt.xticks([r + bar_width/2 for r in range(len(sorted_results))], sorted_results['Features'], rotation=45, ha="right")
    
    # Adding legend
    plt.legend()
    
    # Show the plot
    plt.show()

#%%

def plot_r2adjusted_bar_chart(model_results):
    # Sort the DataFrame based on Test R2_Adjusted
    sorted_results = model_results.sort_values(by='Test_R2_Adjusted', ascending= False)
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the figure size
    plt.figure(figsize=(12, 6))
    
    # Set the positions of bars on X-axis
    r1 = range(len(sorted_results))
    r2 = [x + bar_width for x in r1]
    
    # Plotting the bars
    plt.bar(r1, sorted_results['Train_R2_Adjusted'], width=bar_width, edgecolor='grey', label='Train R2_Adjusted')
    plt.bar(r2, sorted_results['Test_R2_Adjusted'], width=bar_width, edgecolor='grey', label='Test R2_Adjusted')
    
    # Adding labels
    plt.xlabel('Feature Combinations', fontweight='bold')
    plt.ylabel('R2_Adjusted', fontweight='bold')
    
    # Set X-axis ticks and labels
    plt.xticks([r + bar_width/2 for r in range(len(sorted_results))], sorted_results['Features'], rotation=45, ha="right")
    
    # Adding legend
    plt.legend()
    
    # Show the plot
    plt.show()

#%%

