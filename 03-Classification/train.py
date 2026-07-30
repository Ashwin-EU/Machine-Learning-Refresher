# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle

# Parameters
C = 1
n_splits = 10

# Output file
output_file = f'model_C={C}.bin'

# Data preparaion
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/refs/heads/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'Yes').astype(int)
numeric = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod']
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Model Training
def train(df, y_train, C):
    dicts = df[categorical + numeric].to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C = C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numeric].to_dict(orient = 'records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred

# Model Validation
print(f'Starting Cross Validation with C={C}')
fold = 0
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    dv, model = train(df_train, y_train, C)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1
print('Cross Validation Result')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# Training the Final Model
print('Training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# Save Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f'Model is saved to {output_file}')



