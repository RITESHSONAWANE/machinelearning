# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:07:46 2020

@author: RITESH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score

df=pd.read_csv('heart.csv')
df.info()

df.describe()

corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.hist()
sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')

#data Processing

heart=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

standardScaler=StandardScaler()
columns_to_scale=['age','trestbps','chol','thalach','oldpeak']
heart[columns_to_scale]=standardScaler.fit_transform(heart[columns_to_scale])
heart.head()

y=heart['target']
X=heart.drop(['target'],axis=1)
knn_scores=[]
for k in range(1,21):
    knn_classifier=KNeighborsClassifier(n_neighbors=12)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())
    
plt.plot([k for k in range(1,21)],knn_scores,color='red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))
    plt.xticks([i for i in range(1,21)])
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Scores')
    plt.title('K neighbors classifier scores for different k values')
    
    
    
    score.mean()



