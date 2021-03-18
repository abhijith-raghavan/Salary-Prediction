# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('data/salarydata.csv')

#Feature Engineering
data['marital-status'].replace(['Separated'], 'Separated',inplace=True)
data['marital-status'].replace(['Divorced'], 'Separated',inplace=True)

#Label Encoding Data
label_encoder = LabelEncoder()
for i in ['workclass', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex','native-country', 'salary']:
    data[i] = label_encoder.fit_transform(data[i])
    
X=data.drop(['salary'],axis=1)
y=pd.DataFrame(data['salary'])

sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

gb=GradientBoostingClassifier(max_depth=6, min_samples_split=2, min_samples_leaf=1, subsample=1,random_state=42)
gb.fit(X_train,y_train.values.ravel())
y_pred=gb.predict(X_test)

pickle.dump(gb, open('data/model.pkl','wb'))



