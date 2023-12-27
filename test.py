import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    # Load data from Excel file
    data = pd.read_csv("./Data.csv")
    return data

def preprocess_data(data):
    # Convert categorical variables to numerical using Label Encoding
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Drop unnecessary columns
    data = data.drop(['ID', 'Dt_Customer'], axis=1)

    # Split the data into features and labels
    X = data.drop('Response', axis=1)
    y = data['Response']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def svm_train_predict(X_train, X_test, y_train, kernel='linear'):
    # Train SVM model
    from sklearn.svm import SVC
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    return y_pred

def evaluate(y_true, y_pred):
    # Evaluate the performance of the model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, report, matrix

if __name__ == "__main__":
    # Load data
    file_path = "path/to/your/file.xlsx"
    data = load_data(file_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train and predict using SVM
    y_pred = svm_train_predict(X_train, X_test, y_train)

    # Evaluate the model
    accuracy, classification_report, confusion_matrix = evaluate(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report)
    print("Confusion Matrix:")
    print(confusion_matrix)
