import pandas as pd
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data_path = r'\data\processed_KDDTest.csv'
data = pd.read_csv(data_path)

# Preprocess the data
def preprocess_data(data):
    # Encode the target variable
    data['binary_label'] = data['label'].apply(lambda x: 1 if x != 'normal' else 0)

    # Select relevant features based on your requirements
    features = data[['num_failed_logins', 'su_attempted', 'srv_serror_rate', 
                     'src_bytes', 'dst_bytes', 'binary_label']].copy()  # Make a copy to avoid warnings

    # Set conditions for low priority traffic
    low_priority_conditions = (features['num_failed_logins'] > 0) | \
                              (features['su_attempted'] > 0) | \
                              (features['srv_serror_rate'] > 0)

    # Assign low priority to these conditions using .loc
    features.loc[:, 'low_priority'] = low_priority_conditions.astype(int)

    # Simulate rerouting traffic by adjusting src_bytes and dst_bytes
    # Use .loc to avoid SettingWithCopyWarning
    features.loc[:, 'optimized_src_bytes'] = features['src_bytes'].where(features['low_priority'] == 0, features['src_bytes'] * 0.5)
    features.loc[:, 'optimized_dst_bytes'] = features['dst_bytes'].where(features['low_priority'] == 0, features['dst_bytes'] * 0.5)

    # Prepare feature matrix X and target vector y
    X = features[['num_failed_logins', 'su_attempted', 'srv_serror_rate', 
                  'optimized_src_bytes', 'optimized_dst_bytes']]
    y = features['binary_label']

    return X, y, features

# Run preprocessing
X, y, features = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Support Vector Classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
