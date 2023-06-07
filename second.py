import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the training data
data = pd.read_csv('TrainingDataMulti.csv',header=None)
features = data.iloc[:, :128]  # To Select all rows and columns 1-128
labels = data.iloc[:, 128]    # To Select all rows and only the column 129

# Split the data into 85% training set and 15% validation set
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.15, random_state=30)

# Train the model
model = RandomForestClassifier()
# model=SVC()
# model=DecisionTreeClassifier()
# model=LogisticRegression()
# model=KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions on the Training set
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred,average='weighted')
train_recall = recall_score(y_train, y_train_pred,average='weighted')
train_f1_score = f1_score(y_train, y_train_pred,average='weighted')

# Make predictions on the validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred,average='weighted')
val_recall = recall_score(y_val, y_val_pred,average='weighted')
val_f1_score = f1_score(y_val, y_val_pred,average='weighted')

# Print the performance metrics on training and validation sets
print("Training Accuracy:", train_accuracy)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1 Score:", train_f1_score)
print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1_score)

testing_data = pd.read_csv('TestingDataMulti.csv', header=None)

testing_features = testing_data.iloc[:, :128]


# Preprocess the testing data (apply the same preprocessing steps as the training data)

# Make predictions on the testing data
testing_predictions = model.predict(testing_features)

# Save the predicted labels for the testing data
testing_data['Label'] = testing_predictions
testing_data.to_csv('TestingResultsMulti.csv', index=False, header=False) 

