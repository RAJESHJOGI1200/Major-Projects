from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Home page view
def home(request):
    return render(request, 'home.html')

# Prediction page view
def predict(request):
    return render(request, 'predict.html')

# Result page view where the prediction logic is implemented
def result(request):
    # Load the dataset
    data = pd.read_csv(r'C:\Users\DELL\Desktop\Projects\Diabetes_Prediction\diabetes.csv')

    # Prepare features and target variable
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

    # Define parameter grid for RandomizedSearchCV
    param_dist_rf = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Initialize RandomizedSearchCV
    random_search_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=1),
        param_dist_rf,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=1,
        verbose=2
    )

    # Fit the model using the best parameters found
    random_search_rf.fit(X_train, y_train)

    # Display the best parameters
    print(f"Best Parameters for Random Forest: {random_search_rf.best_params_}")

    # Train the model with the best parameters
    best_rf_model = random_search_rf.best_estimator_

    # Evaluate the model on the test set
    y_pred_rf = best_rf_model.predict(X_test)
    best_rf_accuracy = accuracy_score(y_test, y_pred_rf)

    # Print evaluation metrics for debugging purposes
    print(f"Optimized Random Forest Accuracy: {best_rf_accuracy * 100:.2f}%")
    print("Confusion Matrix on Test Data:\n", confusion_matrix(y_test, y_pred_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Retrieve input values from request
    try:
        val1 = float(request.GET['n1'])
        val2 = float(request.GET['n2'])
        val3 = float(request.GET['n3'])
        val4 = float(request.GET['n4'])
        val5 = float(request.GET['n5'])
        val6 = float(request.GET['n6'])
        val7 = float(request.GET['n7'])
        val8 = float(request.GET['n8'])
    except ValueError as e:
        print(f"Error parsing input values: {e}")
        return render(request, 'predict.html', {"result2": "Invalid input"})

    # Standardize the input values
    input_scaled = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Make a prediction using the trained model
    pred = best_rf_model.predict(input_scaled)

    # Print the prediction for debugging
    print(f"Prediction for input {input_scaled}: {pred}")

    # Display result based on prediction
    result1 = "Positive" if pred[0] == 1 else "Negative"

    # Return result to the template
    return render(request, 'predict.html', {"result2": result1})


