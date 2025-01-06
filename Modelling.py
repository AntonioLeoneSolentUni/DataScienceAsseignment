import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

df = pd.read_csv('CleanedDataset.csv')


ind_drop = df[df['Question'].apply(lambda x: x.startswith('Percentage of adults who are blind or unable to see at all (NHIS Adult Module)'))].index
df = df.drop(ind_drop)


ind_drop = df[df['Question'].apply(lambda x: x.startswith('Percentage of adults who even when wearing glasses or contact lenses find it difficult to find something on a crowded shelf (NHIS Adult Module)'))].index
df = df.drop(ind_drop)

ind_drop = df[df['Question'].apply(lambda x: x.startswith('Percentage of adults who are blind or unable to see at all (NHIS Adult Module)'))].index
df = df.drop(ind_drop)

ind_drop = df[df['Question'].apply(lambda x: x.startswith(  'Percentage of people who wear glasses (NHIS Functioning and Disability Module)'
))].index
df = df.drop(ind_drop)

ind_drop = df[df['Question'].apply(lambda x: x.startswith(   'Percentage of adults who even when wearing glasses or contact lenses find it difficult to notice objects off to the side (NHIS Adult Module)'
))].index
df = df.drop(ind_drop)

ind_drop = df[df['Question'].apply(lambda x: x.startswith(   'Percentage of adults who even when wearing glasses or contact lenses find it difficult to go down steps, stairs, or curbs in dim light or at night (NHIS Adult Module)'
))].index
df = df.drop(ind_drop)




# Feature and Target
#Features: Age, Gender, RiskFactor, RiskFactorResponse, Sample Size (weighting)
X = df[['Age', 'Gender', 'RiskFactor', 'RiskFactorResponse']]
y = df['Response']

label_encoder = LabelEncoder()
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # Store the encoder for future use
 # Encode features
y_encoded = label_encoder.fit_transform(y)    # Encode target


# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features (important for algorithms like SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
dt_pred = dt_model.predict(X_test_scaled)
print(f"\nDecision Tree Accuracy: {accuracy_score(y_test, dt_pred)}")  # Expected: Accuracy score



# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_pred = rf_model.predict(X_test_scaled)
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, rf_pred)}")  # Expected: Accuracy score



# Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', random_state=42)  # Use 'linear' kernel
svm_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
svm_pred = svm_model.predict(X_test_scaled)
print(f"\nSVM Accuracy: {accuracy_score(y_test, svm_pred)}")  # Expected: Accuracy score



# K-Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust k
knn_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
knn_pred = knn_model.predict(X_test_scaled)
print(f"\nKNN Accuracy: {accuracy_score(y_test, knn_pred)}")  # Expected: Accuracy score



# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
nb_pred = nb_model.predict(X_test_scaled)
print(f"\nNaive Bayes Accuracy: {accuracy_score(y_test, nb_pred)}")  # Expected: Accuracy score



# Linear Regression (for continuous outcomes, could be cancer progression score or survival time)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)  # Here y_train would be continuous in this case

# Predict and Evaluate
linear_pred = linear_model.predict(X_test_scaled)
print(f"\nLinear Regression R^2: {r2_score(y_test, linear_pred)}")  # Expected: R^2 score



# Decision Tree Regressor (for continuous outcomes)
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)  # Here y_train would be continuous

# Predict and Evaluate
dt_regressor_pred = dt_regressor.predict(X_test_scaled)
print(f"\nDecision Tree Regressor R^2: {r2_score(y_test, dt_regressor_pred)}")  # Expected: R^2 score



# Random Forest Regressor (for continuous outcomes)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Predict and Evaluate
rf_regressor_pred = rf_regressor.predict(X_test_scaled)
print(f"\nRandom Forest Regressor R^2: {r2_score(y_test, rf_regressor_pred)}")  # Expected: R^2 score



# Support Vector Regression (SVR)
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)  # Again, y_train should be continuous for regression

# Predict and Evaluate
svr_pred = svr_model.predict(X_test_scaled)
print(f"\nSVR R^2: {r2_score(y_test, svr_pred)}")  # Expected: R^2 score



# K-Means Clustering (for grouping similar cancer types, for example)
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters for benign/malignant (for example)
kmeans.fit(X_train_scaled)  # Note: no target labels, just the features

# Print cluster centers
print(f"\nK-Means Cluster Centers: {kmeans.cluster_centers_}")

# Predict clusters
kmeans_pred = kmeans.predict(X_test_scaled)
print(f"Clusters predicted by KMeans: {np.unique(kmeans_pred)}")  # Expected: 2 clusters (benign and malignant)


# **Model Evaluation Summary:**
print("\nModel Evaluation Summary:")

# **Classification Models (Accuracy)**
print(f"Accuracy (Decision Tree): {accuracy_score(y_test, dt_pred)}")
print(f"Accuracy (Random Forest): {accuracy_score(y_test, rf_pred)}")
print(f"Accuracy (SVM): {accuracy_score(y_test, svm_pred)}")
print(f"Accuracy (KNN): {accuracy_score(y_test, knn_pred)}")
print(f"Accuracy (Naive Bayes): {accuracy_score(y_test, nb_pred)}")

# **Regression Models (R^2)**
print(f"R^2 (Linear Regression): {r2_score(y_test, linear_pred)}")
print(f"R^2 (Decision Tree Regressor): {r2_score(y_test, dt_regressor_pred)}")
print(f"R^2 (Random Forest Regressor): {r2_score(y_test, rf_regressor_pred)}")
print(f"R^2 (SVR): {r2_score(y_test, svr_pred)}")
