import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Step 1: Load the Iris dataset from the provided URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(url)

# Step 2: Split the dataset into training and testing datasets
Train_Data, Test_Data = train_test_split(iris_df, test_size=0.2, random_state=42)

# Step 3: Save these datasets as CSV files
Train_Data.to_csv('IrisTest_TrainData.csv', index=False)
Test_Data.to_csv('IrisTest_TestData.csv', index=False)

# Step 4: Perform the analysis and model evaluations
# 1. How many missing values are there in Train_Data?
missing_train = Train_Data.isnull().sum().sum()

# 2. What is the proportion of Setosa types in Test_Data?
setosa_proportion = Test_Data[Test_Data['species'] == 'setosa'].shape[0] / Test_Data.shape[0]

# 3. Build and evaluate K-Nearest Neighbors (KNN) model (model_1)
knn_model = KNeighborsClassifier(n_neighbors=3)
X_train = Train_Data.drop('species', axis=1)
y_train = Train_Data['species']
X_test = Test_Data.drop('species', axis=1)
y_test = Test_Data['species']

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# Calculate accuracy score of KNN model (model_1)
accuracy_model_1 = accuracy_score(y_test, y_pred)

# Identify misclassified samples from model_1
misclassified_indices = Test_Data.index[y_test != y_pred].tolist()

# 4. Build and evaluate Logistic Regression model (model_2)
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)

# Calculate accuracy of Logistic Regression model (model_2)
accuracy_model_2 = accuracy_score(y_test, y_pred_logreg)

# Print results
print(f"1. Number of missing values in Train_Data: {missing_train}")
print(f"2. Proportion of Setosa types in Test_Data: {setosa_proportion:.2f}")
print(f"3. Accuracy score of K-Nearest Neighbor model (model_1): {accuracy_model_1:.2f}")
print(f"4. Indices of misclassified samples from model_1: {misclassified_indices}")
print(f"5. Accuracy of Logistic Regression model (model_2): {accuracy_model_2:.2f}")

# Step 5: Plotting the data
# Pairplot
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'X'], palette='Set1')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Boxplot of Sepal Length by Species
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='sepal_length', data=iris_df)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

# Violinplot of Petal Width by Species
plt.figure(figsize=(8, 6))
sns.violinplot(x='species', y='petal_width', data=iris_df)
plt.title('Violinplot of Petal Width by Species')
plt.show()
