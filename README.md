# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
## AIM:
To predict house prices using regression models and compare the performance of different machine learning regression models based on RMSE, MAE, and R².
1.Machine Learning:Machine Learning is used to learn patterns from existing data and make predictions. 
•	Regression is a supervised learning technique used to predict continuous numerical values. 
•	In this experiment, regression models are used to predict the price of a house. 
•	The dataset contains house-related features such as: 
o	square_feet 
o	num_rooms 
o	age 
o	distance_to_city(km) 
•	The target variable is: 
o	price 
## DATASET DESCRIPTION
•	Dataset: House Price Dataset 
•	Problem: Predict house price. 
•	Features (X): 
o	square_feet – size of the house. 
o	num_rooms – number of rooms. 
o	age – age of the house in years. 
o	distance_to_city(km) – distance from the city centre. 
•	Target (y): 
o	price – continuous house price. 
## PROBLEM STATEMENT
•	Develop a machine learning model to predict house prices. 
•	Use house characteristics as input. 
•	Train different regression models. 
•	Compare their prediction performance. 
•	Select the better-performing model based on evaluation metrics. 
## REGRESSION MODELS USED
The uploaded notebook compares the following models:
1.	Linear Regression 
2.	Ridge Regression 
3.	Lasso Regression 
4.	ElasticNet Regression 
5.	Polynomial Regression 
6.	Decision Tree Regressor 
7.	Random Forest Regressor 
8.	Gradient Boosting Regressor 
9.	Support Vector Regressor (SVR) 
10.	K-Nearest Neighbors (KNN) Regressor 
### LIBRARIES USED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
### LOAD THE DATASET
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

df.head()
### DATA OVERVIEW
Display the dataset
df
Display information
df.info()
Display shape
df.shape
Summary statistics
df.describe()
Check missing values
df.isnull().sum()
## EXPLORATORY DATA ANALYSIS
Distribution of Features
The notebook examines the distribution of:
•	square_feet 
•	num_rooms 
•	age 
•	distance_to_city(km) 
•	price 
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
9. CORRELATION ANALYSIS
•	Correlation shows the relationship between numerical variables. 
•	A correlation heatmap is used to visualize these relationships. 
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()
10. SCATTER PLOTS
Scatter plots are used to study the relationship between individual features and house price.
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.show()
## OUTLIER DETECTION
•	Boxplots are used to identify extreme values. 
•	Outliers may negatively affect regression models. 
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
## OUTLIER TREATMENT
The notebook removes extremely low and extremely high house prices using the 1st and 99th percentiles.
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]
•	This reduces the effect of extreme house prices. 
•	It helps the models learn from more typical observations. 
13. DEFINE FEATURES AND TARGET
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']
•	X → Input features. 
•	y → Target house price. 
14. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
•	80% → Training data 
•	20% → Testing data 
## FEATURE SCALING
The notebook uses StandardScaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
•	Scaling puts the features on a comparable scale. 
•	The scaler is fitted only on training data. 
•	The same transformation is applied to test data. 

## BASELINE MODEL
The baseline predicts the mean house price for every test observation.
y_pred_baseline = (
    np.mean(y_train) *
    np.ones_like(y_test)
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline RMSE: {rmse_baseline:.2f}, "
    f"MAE: {mae_baseline:.2f}"
)

## LINEAR REGRESSION
•	Linear Regression finds a linear relationship between input features and house price. 
•	It is used as the main baseline regression model. 
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

## RIDGE REGRESSION
•	Ridge Regression is a regularized version of Linear Regression. 
•	It helps control large model coefficients. 
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

## LASSO REGRESSION
•	Lasso Regression uses L1 regularization. 
•	It can reduce some feature coefficients toward zero. 
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

##  ELASTIC NET REGRESSION
•	ElasticNet combines L1 and L2 regularization. 
•	It is useful when several features may contribute to the prediction. 
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

##  POLYNOMIAL REGRESSION
•	Polynomial Regression extends Linear Regression by creating polynomial features. 
•	The notebook uses degree 2. 
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)

##  DECISION TREE REGRESSOR
•	A Decision Tree divides the data into different regions based on feature values. 
•	It can model nonlinear relationships. 
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)

##  RANDOM FOREST REGRESSOR
•	Random Forest combines multiple decision trees. 
•	It generally provides more robust predictions than a single tree. 
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

##  GRADIENT BOOSTING REGRESSOR
•	Gradient Boosting builds models sequentially. 
•	Each new model attempts to improve the errors of previous models. 
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)

## SUPPORT VECTOR REGRESSOR
•	SVR uses Support Vector Machine principles for regression. 
•	The notebook uses an RBF kernel. 
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

##  KNN REGRESSOR
•	KNN predicts a value based on nearby observations. 
•	The notebook uses 5 neighbors. 
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

## MODEL EVALUATION
The notebook uses three metrics:
RMSE
•	Root Mean Squared Error. 
•	Lower value indicates better performance. 
MAE
•	Mean Absolute Error. 
•	Lower value indicates better performance. 
R²
•	Measures how well the model explains variation in house prices. 
•	Higher value generally indicates better performance. 
Code
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.sort_values(
    by="RMSE"
)

##  MODEL COMPARISON
<img width="601" height="250" alt="image" src="https://github.com/user-attachments/assets/bca8dfef-1052-4e92-90e1-0f6356888be5" />
		

### Comparison Criteria
•	Lower RMSE → Better model. 
•	Lower MAE → Better model. 
•	Higher R² → Better model. 
The notebook sorts the models according to RMSE to compare their performance.
## ACTUAL VS PREDICTED PRICE
plt.figure(figsize=(15,12))

for i, (name, y_pred) in enumerate(models.items()):

    plt.subplot(5,2,i+1)

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")

plt.tight_layout()
plt.show()
•	The plot compares actual house prices with predicted prices. 
•	Points closer to the diagonal line indicate better predictions. 
30. RESIDUAL ANALYSIS
•	A residual is the difference between actual and predicted values. 
Residual = Actual Price − Predicted Price
for name, y_pred in models.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color='r',
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.show()
•	Residuals close to zero indicate smaller prediction errors. 
•	Residual plots help identify unusual prediction patterns. 
##  RANDOM FOREST FEATURE IMPORTANCE
The notebook calculates the importance of each feature using Random Forest.
importances = rf.feature_importances_

feat_names = X.columns

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances,
    y=feat_names
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()
This helps identify which house features contribute more to the Random Forest prediction.
##  GRADIENT BOOSTING FEATURE IMPORTANCE
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances_gbr,
    y=feat_names
)

plt.title(  "Gradient Boosting Feature Importance")

plt.show()
This shows the relative contribution of the house features to the Gradient Boosting model.

## RMSE COMPARISON GRAPH
plt.figure(figsize=(10,6))

sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE")
)

plt.title(
    "RMSE Comparison Across Regression Models"
)

plt.show()
•	The graph provides a visual comparison of model errors. 
•	The model with the lowest RMSE performs best according to this metric.
## OUTPUT
<img width="705" height="615" alt="683399103-db43ccca-931f-4ce6-96ab-746765b0f81c" src="https://github.com/user-attachments/assets/f7f1c4e6-eed4-4072-ac7d-d650fccc8cf2" />
<img width="762" height="747" alt="683399145-d7d29efc-108e-4cd6-ac2e-a1a01c0d4ba2" src="https://github.com/user-attachments/assets/544b6cd6-6f35-40ee-bd24-e98ce43df51c" />
<img width="762" height="747" alt="683399145-d7d29efc-108e-4cd6-ac2e-a1a01c0d4ba2" src="https://github.com/user-attachments/assets/98177288-b4cb-47ec-931b-3864402d03ba" />
<img width="762" height="747" alt="683399260-0d4baf48-0154-4994-92d2-aba60a4c2692" src="https://github.com/user-attachments/assets/a9667af8-7e39-46be-b350-7428a34abbf7" />
<img width="762" height="747" alt="683399260-0d4baf48-0154-4994-92d2-aba60a4c2692" src="https://github.com/user-attachments/assets/bdd38537-b821-4754-97d1-6c7ec008257f" />
<img width="762" height="747" alt="683399408-5e3fca36-cd64-4d6a-a891-4a6bb6226d6c" src="https://github.com/user-attachments/assets/20a8a02e-27ca-4820-a6b9-e3eb7600fd0f" />
<img width="762" height="747" alt="683399332-36f891ed-adbb-4adf-918f-0141a062ebd5" src="https://github.com/user-attachments/assets/a1db0315-42bd-47d4-9d96-08daa715abfc" />
<img width="762" height="747" alt="683399408-5e3fca36-cd64-4d6a-a891-4a6bb6226d6c" src="https://github.com/user-attachments/assets/99fefbc9-d1ff-45e5-8453-47f4d0ae56e9" />
<img width="725" height="375" alt="683399470-9626aff9-5377-43d1-9152-6e0fffb90956" src="https://github.com/user-attachments/assets/79905a05-8419-4967-8ae1-2cb612912559" />
<img width="761" height="682" alt="683399518-a9092318-21b4-42c8-b156-f8ee89328c38" src="https://github.com/user-attachments/assets/5f0e7437-a9b7-43a1-94c2-255d0d3f1943" />
<img width="761" height="682" alt="683399518-a9092318-21b4-42c8-b156-f8ee89328c38" src="https://github.com/user-attachments/assets/4634a2c1-6e26-4c55-b7e6-2c6a2cc394a0" />
<img width="726" height="325" alt="683399586-d0859043-a438-4874-b34c-3d7506d63e21" src="https://github.com/user-attachments/assets/d13784f1-445d-4929-a60e-b00242f2f08e" />
<img width="802" height="647" alt="683399640-06c467cb-0425-4ff6-815d-6d881c0d36b3" src="https://github.com/user-attachments/assets/f18037f8-73c7-40e4-bfc2-078c4f1c930e" />
<img width="756" height="497" alt="683399816-9e9cdcb1-ea47-4647-a055-7d22f409a367" src="https://github.com/user-attachments/assets/b5ae5d85-32ee-406c-a127-d66e1407e1c8" />
<img width="785" height="492" alt="683399879-67065470-79b9-4e22-aef7-baf6cd6f4bf0" src="https://github.com/user-attachments/assets/a5d013e3-df7b-409c-8fcd-16ae0cd8a59a" />
<img width="777" height="482" alt="683399986-09af90ab-de6b-49cf-9375-de8340b7c663" src="https://github.com/user-attachments/assets/e892e211-1fc1-47cf-8a61-c8ba972d5a5f" />
<img width="730" height="750" alt="683400062-58d977fe-e1a9-40ee-9848-8219685c0b1b" src="https://github.com/user-attachments/assets/eb122fac-dfa4-48ed-9a1b-b5376f900f86" />
<img width="837" height="802" alt="683400142-745ecec1-2179-4e92-8111-f7764fb97627" src="https://github.com/user-attachments/assets/b6967f6d-68d5-4be9-ac87-83981749e713" />
<img width="792" height="517" alt="683400202-368118a3-8941-4785-b59f-1b796ccc58ab" src="https://github.com/user-attachments/assets/2a808ff6-b1d0-45b7-ab82-9d885a2f94b4" />
<img width="792" height="517" alt="683400252-de5beb8f-0989-48db-89a3-9590d05ce58f" src="https://github.com/user-attachments/assets/55b82c8a-03f2-49a7-9b0c-eb6a99b356f7" />
<img width="802" height="542" alt="683400290-a67d52fd-31ea-4063-a769-897ed1ed2f43" src="https://github.com/user-attachments/assets/842e91d2-db3a-49ea-814f-ccae6a896ba6" />
<img width="735" height="492" alt="683400330-c450a1ad-be3d-4411-8916-cd3d36624872" src="https://github.com/user-attachments/assets/53d79b90-ce27-4aa2-91f2-40c199403689" />
<img width="727" height="462" alt="683400377-5a528aec-a0d9-47c5-8b7e-2401cf1dc4d9" src="https://github.com/user-attachments/assets/a3c0b517-1d70-49de-b197-9734d6dde9ef" />
<img width="722" height="502" alt="683400462-69a2d638-9f42-4e49-83ab-2f43d5a469e5" src="https://github.com/user-attachments/assets/646553f6-db1f-4471-ba8c-12bbc2f2ff04" />
<img width="727" height="497" alt="683400508-951c09ca-11a8-4c03-935b-9c51554c6480" src="https://github.com/user-attachments/assets/0d53c498-fdcd-4409-9c12-1e54ee94407c" />
<img width="732" height="496" alt="683400574-3781cfa7-8c1e-452a-904c-a0108f5f5ae3" src="https://github.com/user-attachments/assets/e3075a0f-544a-4e2e-b76b-951dbd8b9b18" />
<img width="736" height="477" alt="683400634-a9477858-0e71-47fd-9e76-0e7267537fbb" src="https://github.com/user-attachments/assets/e78c3377-6a64-438c-b5da-c72e4042ec67" />
<img width="717" height="481" alt="683400680-f5fe85ad-5f20-4ff8-831d-38688db892d6" src="https://github.com/user-attachments/assets/03fad011-dc30-4ae7-ae9a-5acf9352c70d" />
<img width="711" height="472" alt="683400723-ce76d264-e7b2-4c9f-b1d2-759204ab6603" src="https://github.com/user-attachments/assets/9e60dcf5-f0ef-4472-a2d1-79f7d8f68354" />
<img width="722" height="500" alt="683400746-f70cd8da-ccee-4de9-8c70-898b2d58e6d1" src="https://github.com/user-attachments/assets/95ef58f2-ba99-479a-92e1-bcca87d07e7d" />

## CONCLUSION
Thus, Linear Regression and other regression models were successfully applied for house price prediction, and their performance was compared using standard regression evaluation metrics.

