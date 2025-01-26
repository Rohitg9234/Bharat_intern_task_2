## README for Wine Quality Prediction Project

This project aims to predict the quality of wine based on various chemical attributes using machine learning techniques. The goal is to create a model that classifies wine quality into two categories: "good" or "bad". The quality of the wine is evaluated using machine learning algorithms like Random Forest Classifier, Support Vector Classifier (SVC), and Stochastic Gradient Descent (SGD). 

### Overview
In this project, we use the **Wine Quality dataset** to build models that predict the quality of wine. The dataset consists of several chemical properties of red wine, and the target variable is the **quality** of the wine, rated on a scale from 0 to 10.

### Dataset
The dataset contains the following columns:
- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **sulphates**
- **alcohol**
- **quality** (target variable, which is categorized into “bad” or “good” in the model)

### Steps in the Code

#### 1. **Data Loading and Inspection**
The dataset is loaded using `pandas.read_csv()` from a CSV file. We first inspect the dataset using `head()` to get a preview and `info()` to understand the data types of each column.

```python
wine = pd.read_csv('/content/winequality-red.csv')
wine.head()
wine.info()
```

#### 2. **Exploratory Data Analysis (EDA)**
Several bar plots are created to analyze the relationships between wine quality and various chemical properties. For example, `sns.barplot()` is used to visualize how each feature (like **fixed acidity**, **volatile acidity**, **citric acid**, etc.) varies across different wine quality levels.

```python
fig = plt.figure(figsize=(10,6))
sns.barplot(x='quality', y='fixed acidity', data=wine)
```

#### 3. **Data Preprocessing**
- **Binary Classification**: The quality of wine is converted into a binary classification problem where wines with a quality above 6.5 are labeled "good" and the rest as "bad".
- **Label Encoding**: The categorical values of "good" and "bad" are encoded into numeric labels (0 for "bad" and 1 for "good").

```python
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
```

#### 4. **Feature and Target Variable Separation**
The features (input variables) and target (quality) are separated into `X` and `y` respectively.

```python
X = wine.drop('quality', axis=1)
y = wine['quality']
```

#### 5. **Train-Test Split**
The data is split into training and testing sets using `train_test_split()`, where 20% of the data is reserved for testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 6. **Standardization of Features**
To optimize performance, the features are standardized using `StandardScaler()`. This scales the data so that each feature has a mean of 0 and a standard deviation of 1.

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
```

#### 7. **Model Training and Evaluation**

##### 7.1 **Random Forest Classifier**
A Random Forest Classifier is trained using the training data. It is then evaluated on the test data. The `classification_report()` gives metrics like precision, recall, and F1-score, while the `confusion_matrix()` shows the number of correct and incorrect predictions.

```python
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
```

##### 7.2 **Stochastic Gradient Descent (SGD)**
The Stochastic Gradient Descent classifier is trained similarly to Random Forest and evaluated using the same methods.

```python
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))
```

##### 7.3 **Support Vector Classifier (SVC)**
An SVC model is also trained and evaluated. We then use **GridSearchCV** to fine-tune hyperparameters like `C`, `gamma`, and `kernel` to improve the performance.

```python
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))

param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
```

After hyperparameter tuning, the model’s accuracy increases from 86% to 90%.

```python
svc2 = SVC(C=1.2, gamma=0.9, kernel='rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
```

#### 8. **Cross-Validation**
Finally, cross-validation is used to evaluate the Random Forest model on different subsets of the data to get a more reliable estimate of its performance.

```python
rfc_eval = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
print(rfc_eval.mean())
```

### Results
- **Random Forest Classifier**: Achieves an accuracy of **87%**.
- **Stochastic Gradient Descent**: Achieves an accuracy of **84%**.
- **Support Vector Classifier (SVC)**: Achieves an accuracy of **86%**, and after hyperparameter tuning, it improves to **90%**.

### Installation Instructions
To run this project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   ```

2. **Install required libraries**:
   Make sure you have Python 3.x installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Download the **Wine Quality dataset** and place it in the `data` folder. Alternatively, you can upload it to the appropriate directory if running on a Jupyter Notebook environment.

4. **Run the Notebook**:
   Open the `prediction-of-quality-of-wine.ipynb` file in Jupyter Notebook or Google Colab and run all the cells.

### Conclusion
This project demonstrates the application of machine learning algorithms for predicting the quality of wine based on its chemical properties. By comparing different models and performing hyperparameter tuning, we achieve a high-performing model capable of classifying wine quality effectively.


