import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

df = pd.read_csv(
    r"C:\Users\dell\OneDrive\Desktop\Credit Risk Modelling\dataset\german_credit_data.csv"
)

df.head()
df.shape
df.info()
df.describe()
df.columns
df["Unnamed: 0"].all

df.drop(columns="Unnamed: 0", axis=1, inplace=True)
df.head()

df["Saving accounts"].isnull().sum()
df["Checking account"].isna().sum()

num_cols = df.select_dtypes(include=["number"])
cat_cols = df.select_dtypes(include=["object"])

num_cols
cat_cols

num_cols.corr()
sns.heatmap(num_cols.corr(), cmap="coolwarm", annot=True, fmt=".2f")

df["Credit amount"].describe()

df["Duration"].describe()

sns.scatterplot(x=df["Credit amount"], y=df["Duration"], hue=df["Sex"])

df.isnull().sum()

df["Saving accounts"].isnull().sum()

df["Checking account"].isnull().sum()

imputer = SimpleImputer(strategy="most_frequent")

df["Saving accounts"] = imputer.fit_transform(df[["Saving accounts"]]).ravel()

df["Saving accounts"].isnull().sum()

df["Checking account"] = imputer.fit_transform(df[["Checking account"]]).ravel()

df["Checking account"].isna().sum()

df.info()

df["Risk"].value_counts()

X = df.drop("Risk", axis=1)
Y = df["Risk"]

df.describe(include="all").T

df[["Age", "Credit amount", "Duration"]].hist(bins=7, edgecolor="black")
plt.suptitle("Distribution Of Numerical Features")
plt.show()

sns.boxplot(df[["Age", "Duration"]], orient="h", whis=2)

df.query("Age > 60").value_counts()

plt.figure(figsize=(8, 8))
for i, col in enumerate(cat_cols):
    plt.subplot(2, 3, i + 1)
    sns.countplot(data=df, x=col, palette="Set2", order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

df.groupby("Job")["Credit amount"].mean()
df.groupby("Sex")["Credit amount"].mean()

sns.scatterplot(
    data=df, x="Age", y="Credit amount", hue="Sex", size="Duration", palette="Set1"
)
plt.title("Credit Amount vs Age color by Sex and sized by Duration")
plt.show()

sns.violinplot(data=df, x="Saving accounts", y="Credit amount", palette="Pastel1")
plt.title("Savings account vs Credit Amount")
plt.show()

df["Risk"].value_counts(normalize=True) * 100

df = df.drop(columns=["Purpose"])
df.head()

cat_cols = df.select_dtypes(include=["object"]).columns.drop("Risk")

num_cols = df.select_dtypes(include=["number"])

target_col = df["Risk"]
target_col

labelEncoder = LabelEncoder()

target_col = labelEncoder.fit_transform(target_col)
target_col
np.unique_counts(target_col)
joblib.dump(labelEncoder, "target_encoder.pkl")

df["Risk"] = target_col
df["Risk"].value_counts()

ohe_dict = {}

ct = ColumnTransformer(
    [("ohe", OneHotEncoder(), cat_cols)],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

df_transformed = ct.fit(X)
joblib.dump(ct, "full_preprocessing.pkl")

df_transformed = pd.DataFrame(df_transformed, columns=ct.get_feature_names_out())
df_transformed.columns.value_counts()
df_transformed

sm = SMOTE(random_state=42)
Y

X_train, X_test, Y_train, Y_test = train_test_split(
    df_transformed, Y, test_size=0.2, random_state=2, stratify=Y
)

X_train_res, y_train_res = sm.fit_resample(X_train, Y_train)
y_train_res.value_counts()


dtc = DecisionTreeClassifier()

dtc.fit(X_train_res, y_train_res)

rfc_pred = dtc.predict(X_test)
accuracy_score(rfc_pred, Y_test)
print(classification_report(rfc_pred, Y_test))

rfc = RandomForestClassifier()

rfc.fit(X_train_res, y_train_res)

rfc_pred = rfc.predict(X_test)
accuracy_score(rfc_pred, Y_test)
print(classification_report(rfc_pred, Y_test))

gbc = GradientBoostingClassifier()
gbc.fit(X_train_res, y_train_res)

rfc_pred = gbc.predict(X_test)
accuracy_score(rfc_pred, Y_test)
print(classification_report(rfc_pred, Y_test))

xgb = XGBClassifier()
xgb.fit(X_train_res, y_train_res)

xgb_pred = xgb.predict(X_test)
accuracy_score(xgb_pred, Y_test)
print(classification_report(rfc_pred, Y_test))

joblib.dump(gbc, "gradient_model.pkl")
