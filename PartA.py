import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

# Load the training data
data = pd.read_csv('TrainingDataBinary.csv', header=None)
 # Select all rows and columns 1-128(Features):
features = data.iloc[:, :128] 
  # Select all rows and only the column 129(Lables):
labels = data.iloc[:, 128]  
# Split the data into 85% training set and 15% validation set:
X_training, X_validation, y_training, y_validation = train_test_split(features, labels, test_size=0.15, random_state=42)
# Scale the input features:
scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_validation_scaled = scaler.transform(X_validation)
# Train the model
model = RandomForestClassifier()
# model=SVC()
# model=DecisionTreeClassifier()
# model=LogisticRegression()
# model=KNeighborsClassifier()
# model = ExtraTreesClassifier()
model.fit(X_training_scaled, y_training)
# Prediction on training Set:
y_training_pred = model.predict(X_training_scaled)
train_accuracy = accuracy_score(y_training, y_training_pred)
train_precision = precision_score(y_training, y_training_pred)
train_f1_score = f1_score(y_training, y_training_pred)

# Make predictions on the validation set
y_validation_pred = model.predict(X_validation_scaled)
val_accuracy = accuracy_score(y_validation, y_validation_pred)
val_precision = precision_score(y_validation, y_validation_pred)
val_f1_score = f1_score(y_validation, y_validation_pred)

# Print the performance metrics on training and validation sets
# print("Training Accuracy:", train_accuracy)
# print("Training Precision:", train_precision)
# print("Training F1 Score:", train_f1_score)
print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation F1 Score:", val_f1_score)


# Represent the result using Confusion Matrix:
confusion_matrix = pd.crosstab(y_validation, y_validation_pred, rownames=['Actual'], colnames=['Predicted'])
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
#Save the entire data to ResultsBinary.csv
testing_data.to_csv('TestingResultsBinary.csv', index=False, header=False)