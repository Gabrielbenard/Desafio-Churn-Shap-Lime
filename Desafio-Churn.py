# !pip install -U ydata-profiling Yellowbrick shap ipywidgets imblearn lime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from ydata_profiling import ProfileReport
from yellowbrick.classifier import ClassificationReport
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
from imblearn.combine import SMOTEENN

#%matplotlib inline


data = pd.read_csv(r"C:\Users\gabri\Documents\Machine Learning\DataSets\churn-A3data.csv")
data

data.info()

data["TotalCharges"].isnull().sum()

profile = ProfileReport(data, title="Pandas Profiling Report")
profile

data['Churn'].value_counts()

plt.hist(data["Churn"],bins=10)
plt.show()

data['Contract'].unique()

data.describe(include=['O'])

df2 = data.copy()
df2['TotalCharges'] = df2['TotalCharges'].str.replace(',', '.', regex=False)
df2['MonthlyCharges'] = df2['MonthlyCharges'].str.replace(',', '.', regex=False)

df2['MonthlyCharges'] = pd.to_numeric(df2['MonthlyCharges'], errors= 'coerce')
df2['TotalCharges'] = pd.to_numeric(df2['TotalCharges'], errors= 'coerce')


imputer = SimpleImputer(strategy='mean')
df2["TotalCharges"] = imputer.fit_transform(df2[["TotalCharges"]])
df2["TotalCharges"].isnull().sum()


X = df2.drop(['customerID','Churn'],axis = 1)
y = df2['Churn'].map({'Yes': 1, 'No': 0})

y.value_counts()

categorical_columns = X.select_dtypes(include =['object']).columns.tolist()
numerical_columns = X.select_dtypes(include =['int64', 'float64']).columns.tolist()

label_encoder = LabelEncoder()
ord_encoder = OrdinalEncoder()
# for col in categorical_columns:
#     X[col] = label_encoder.fit_transform(X[col])
X[categorical_columns] = ord_encoder.fit_transform(X[categorical_columns])

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

len(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled,
                                                    y_resampled,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify = y_resampled)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}\n")

import shap
shap.initjs()

# sample_X = shap.utils.sample(X_train, 200, random_state=42)

# explainer = shap.Explainer(model, sample_X)
# shap_values = explainer(sample_X,check_additivity=False)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)


i = 25  # exemplo específico
shap.plots.force(
    shap_values[i, :, 0],
)

i = 25
shap.plots.force(shap_values[i, :, 1])  # pega apenas a classe 1

i = 20  # exemplo específico
shap.plots.force(shap_values[i, :, 0])

i = 10  # exemplo específico
shap.plots.force(shap_values[i, :, 0])

print("Mapeamento das categorias para cada coluna:")
for i, col in enumerate(categorical_columns):
    print(f"\nColuna '{col}':")
    for j, cat in enumerate(ord_encoder.categories_[i]):
        print(f"  '{cat}' -> {j}")

from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['0','1'], mode='classification')


# Escolher uma instância X_test.iloc[i] para explicar
exp = explainer.explain_instance(X_test.iloc[1].values, model.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True)


from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

importances = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

print(importances)


importances.plot.bar(x="feature", y="importance_mean", yerr="importance_std", legend=False)
plt.title("Permutation Feature Importance")
plt.ylabel("Impacto na performance")
plt.show()

PartialDependenceDisplay.from_estimator(
    model, X_test, features=["tenure","Contract","TotalCharges","MonthlyCharges"], kind="average"
)

#Linha crescente → quanto maior a variável, maior a probabilidade prevista.
#Linha decrescente → efeito contrário.
