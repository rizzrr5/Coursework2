import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load the training data
data = pd.read_csv('TrainingDataBinary.csv', header=None)
features = data.iloc[:, :128]  # Select all rows and columns 1-128
labels = data.iloc[:, 128]    # Select all rows and only the column 129
# Split the data into 85% training set and 15% validation set
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.15, random_state=42)
# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# Train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
# Make predictions on the training set
y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_f1_score = f1_score(y_train, y_train_pred)

# Make predictions on the validation set
y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_f1_score = f1_score(y_val, y_val_pred)

# Print the performance metrics on training and validation sets
# print("Training Accuracy:", train_accuracy)
# print("Training Precision:", train_precision)
# print("Training F1 Score:", train_f1_score)
print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation F1 Score:", val_f1_score)


# Count the occurrences of each class in the training labels
class_counts = labels.value_counts()
confusion_matrix = pd.crosstab(y_val, y_val_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Load the testing data
testing_data = pd.read_csv('TestingDataBinary.csv', header=None)
testing_features = testing_data.iloc[:, :128]

# Scale the testing data
testing_features_scaled = scaler.transform(testing_features)

# Make predictions on the testing data
testing_predictions = model.predict(testing_features_scaled)
print(testing_predictions)

# Save the predicted labels for the testing data
testing_data['Label'] = testing_predictions
testing_data.to_csv('TestingResultsBinary.csv', index=False, header=False)