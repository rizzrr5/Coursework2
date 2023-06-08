import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the training data
data = pd.read_csv('TrainingDataMulti.csv',header=None)
features = data.iloc[:, :128]  # To Select all rows and columns 1-128
labels = data.iloc[:, 128]    # To Select all rows and only the column 129

# Splitting the data into 85% for training set and 15% for validation set
X_training, X_validation, y_training, y_validation = train_test_split(features, labels, test_size=0.15, random_state=42)

# Scaling the input features:
scaler = StandardScaler()
X_training_scaled = scaler.fit_transform(X_training)
X_validation_scaled = scaler.transform(X_validation)

# Training the model
# model = RandomForestClassifier()
# model=SVC()
# model=DecisionTreeClassifier()
# model=LogisticRegression()
# model=KNeighborsClassifier()
model = ExtraTreesClassifier()
model.fit(X_training_scaled, y_training)

# Predictions on the Training set
y_training_pred = model.predict(X_training_scaled) #Not necessary
train_accuracy = accuracy_score(y_training, y_training_pred) #Not necessary
train_precision = precision_score(y_training, y_training_pred,average='weighted') #Not necessary
train_f1_score = f1_score(y_training, y_training_pred,average='weighted') #Not necessary

#Predictions on the validation set
y_validation_pred = model.predict(X_validation_scaled)
val_accuracy = accuracy_score(y_validation, y_validation_pred)
val_precision = precision_score(y_validation, y_validation_pred,average='weighted')
val_f1_score = f1_score(y_validation, y_validation_pred,average='weighted')

#Print Accuracy,precision, F1score
print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation F1 Score:", val_f1_score)

# Represent the result using Confusion Matrix:
confusion_matrix = pd.crosstab(y_validation, y_validation_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

testing_data = pd.read_csv('TestingDataMulti.csv', header=None)

testing_features = testing_data.iloc[:, :128]

testing_features_scaled = scaler.transform(testing_features)

# Making the predictions on the testing data:
testing_predictions = model.predict(testing_features_scaled)

# Save the predicted labels for the testing data
print(testing_predictions)
testing_data['Label'] = testing_predictions
testing_data.to_csv('TestingResultsMulti.csv', index=False, header=False) 

