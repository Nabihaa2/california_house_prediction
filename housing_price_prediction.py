# California Housing Price Prediction Project

# Step 1: Load Dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

housing = fetch_california_housing(as_frame=True).frame

# Step 2: Visualize Data
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()

# Step 3: Stratified Sampling Based on Income Categories
housing["income_cat"] = pd.cut(housing["MedInc"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set = housing.loc[test_idx]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Step 4: Correlation Matrix
corr_matrix = strat_train_set.corr()
print(corr_matrix["MedHouseVal"].sort_values(ascending=False))

# Step 5: Feature Engineering
housing = strat_train_set.copy()
housing["rooms_per_household"] = housing["AveRooms"] / housing["Households"]

# Step 6: Data Preparation Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

housing_features = housing.drop("MedHouseVal", axis=1)
housing_labels = housing["MedHouseVal"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

housing_prepared = pipeline.fit_transform(housing_features)

# Step 7: Train Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lin_reg = LinearRegression().fit(housing_prepared, housing_labels)
tree_reg = DecisionTreeRegressor(random_state=42).fit(housing_prepared, housing_labels)
forest_reg = RandomForestRegressor(random_state=42).fit(housing_prepared, housing_labels)

# Step 8: Evaluation Function
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, name):
    pred = model.predict(housing_prepared)
    rmse = mean_squared_error(housing_labels, pred, squared=False)
    mae = mean_absolute_error(housing_labels, pred)
    print(f"{name}:\n  RMSE: {rmse:.2f}, MAE: {mae:.2f}\n")

evaluate_model(lin_reg, "Linear Regression")
evaluate_model(tree_reg, "Decision Tree")
evaluate_model(forest_reg, "Random Forest")

# Step 9: Visualize Predictions vs Actual
predictions = forest_reg.predict(housing_prepared)
plt.figure(figsize=(8,6))
plt.scatter(housing_labels, predictions, alpha=0.2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regressor: Predicted vs Actual")
plt.plot([0, 5], [0, 5], 'r--')
plt.grid(True)
plt.show()
