import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import seaborn as sns

df=pd.read_csv("E:\creditcard aa.csv")

df.info()

df.describe()

df.isnull().sum().sum()

df['Class'].value_counts()

df1=df[df.Class==0]
df2=df[df.Class==1]

print(df1.shape)

print(df2.shape)

x=df.drop('Class',axis=1)

y=df.Class

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

xtrain

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_scaler=scaler.fit_transform(x)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(xtrain,ytrain)

model.score(xtest,ytest)

model.predict(xtest)
