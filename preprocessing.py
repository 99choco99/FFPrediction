
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # 데이터를 저장하기 위해 사용


fires = pd.read_csv("sanbul2district-divby100.csv")


print(fires.head())
print(fires.info())
print(fires.describe())


print("\n[month value counts]")
print(fires["month"].value_counts())

print("\n[day value counts]")
print(fires["day"].value_counts())


fires["burned_area"] = np.log(fires["burned_area"] + 1)


fires["burned_area"].hist()
plt.title("Burned Area Histogram (log transformed)")
plt.xlabel("log(burned_area + 1)")
plt.ylabel("Frequency")
plt.savefig("burned_area_histogram.png")
plt.clf()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]


scatter_matrix(strat_train_set[["burned_area", "max_temp", "avg_temp", "avg_wind"]], figsize=(10, 8))
plt.savefig("scatter_matrix.png")
plt.clf()


strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                     s=strat_train_set["max_temp"], label="max_temp",
                     c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.savefig("map_plot.png")
plt.clf()


train_data = strat_train_set.drop("burned_area", axis=1)
train_labels = strat_train_set["burned_area"].copy()


num_attribs = train_data.drop(["month", "day"], axis=1).columns.tolist()
cat_attribs = ["month", "day"]


num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])


train_prepared = full_pipeline.fit_transform(train_data)


joblib.dump(train_prepared, "train_prepared.pkl")
joblib.dump(train_labels, "train_labels.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")
