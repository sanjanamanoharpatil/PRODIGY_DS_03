import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
csv_file_path_full = 'bank dataset/bank-full.csv'
bank_full_data = pd.read_csv(csv_file_path_full, delimiter=';')

# Convert categorical variables into numeric using Label Encoding
label_encoders_full = {}
categorical_columns_full = ['job', 'marital', 'education', 'default', 'housing', 
                            'loan', 'contact', 'month', 'poutcome', 'y']

for column in categorical_columns_full:
    le = LabelEncoder()
    bank_full_data[column] = le.fit_transform(bank_full_data[column])
    label_encoders_full[column] = le

# Split the dataset into features (X) and target (y)
X_full = bank_full_data.drop(columns='y')
y_full = bank_full_data['y']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train_full, y_train_full)

# Make predictions on the test set
y_pred_full = clf_full.predict(X_test_full)

# Evaluate the model
accuracy_full = accuracy_score(y_test_full, y_pred_full)
classification_rep_full = classification_report(y_test_full, y_pred_full)

print(f"Accuracy: {accuracy_full}")
print(f"Classification Report:\n{classification_rep_full}")
