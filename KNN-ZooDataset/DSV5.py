import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import requests
from io import StringIO

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/yijiaceline/Machine-Learning-Zoo-Classification/master/zoo.csv"
response = requests.get(url)
data = response.content.decode('utf-8')

# Read the CSV data
df = pd.read_csv(StringIO(data))

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Ensure 'class_type' exists in the DataFrame
if 'class_type' not in df.columns:
    raise ValueError("Column 'class_type' not found in the dataset.")

# Separate features (attributes) and target (class labels)
X = df.drop(['animal_name', 'class_type'], axis=1)  # Features
y = df['class_type']  # Target

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier with a chosen number of neighbors (e.g., k=5)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred, zero_division=1))

# Visualizations

# #  Pairplot for Feature Relationships
# plt.figure(figsize=(12, 8))
# sns.pairplot(df.drop(['animal_name'], axis=1), hue='class_type', palette='viridis')
# plt.suptitle('Pairplot of Zoo Dataset Features', y=1.02)
# plt.show()

#  Correlation Heatmap
corr = df.drop('animal_name', axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()

#  Boxplot of Features by Class Type
plt.figure(figsize=(12, 8))
sns.boxplot(x='class_type', y='legs', data=df, palette='Set3')
plt.title('Distribution of Legs by Class Type')
plt.xlabel('Class Type')
plt.ylabel('Number of Legs')
plt.show()

# Violin Plot for Feature Distributions
plt.figure(figsize=(12, 8))
sns.violinplot(x='class_type', y='legs', data=df, palette='Pastel1')
plt.title('Distribution of Legs by Class Type')
plt.xlabel('Class Type')
plt.ylabel('Number of Legs')
plt.show()

# Count Plot of Class Types
plt.figure(figsize=(8, 6))
sns.countplot(x='class_type', data=df, palette='viridis')
plt.title('Distribution of Class Types')
plt.xlabel('Class Type')
plt.ylabel('Count')
plt.show()
