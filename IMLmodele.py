# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:06:18 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:32:52 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:23:07 2022

@author: Sow Abdoul Aziz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:19:04 2022

@author: aa.sow
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import chdir
chdir (r"C:\Users\Sow Abdoul Aziz")
import pandas as pd
import numpy as np
Base=pd.read_excel("NewBase1.xlsx", index_col="Identifiant de ménage")
Base.head()
Base.dtypes
Base.shape
Base.isnull().sum()
Base.info()
Base["Territoire"].value_counts()
Base.Territoire=Base["Territoire"].fillna("TPM")
Base1_C=Base.drop(["Typologie du menage","Territoire","Age","Ressources"],axis=1)
Base1_Q=Base.drop(["Cible"],axis=1)
Base1_Q.columns
Base1_QD=pd.get_dummies(Base1_Q[["Typologie du menage","Territoire","Age","Ressources"]])
Base1_QD.columns
Base.Territoire=Base["Territoire"].fillna("TPM")

BASE=pd.concat([Base1_QD,Base1_C],axis=1)
BASE.columns
BASE.head()
BASE.dtypes
print(BASE.shape)
BASE.to_csv("BASE.csv",index=True)
Basecsv=pd.read_csv("BASE.csv", index_col="Identifiant de ménage")
Basecsv.head()
BASE.shape
X=BASE.drop(["Cible"],axis=1)
Y=BASE["Cible"]
from sklearn.model_selection import train_test_split
X_app,X_test,Y_app,Y_test=train_test_split(X,Y,test_size=0.33,random_state=5)

######################## RANDOM FOREST   T###############################
from sklearn.ensemble import RandomForestClassifier
# D´efinition des param`etres
forest = RandomForestClassifier(n_estimators=500, min_samples_split=5,oob_score=True)
# Apprentissage
forest = forest.fit(X_app,Y_app)
print(1-forest.oob_score_)
# Erreur de pr´evision sur le test
1-forest.score(X_test,Y_test)
# Pr´evision
Y_pred_rf = forest.predict(X_test)
# Matrice de confusion
mat_conf_rf=pd.crosstab(Y_test, Y_pred_rf)
print(mat_conf_rf)
X_app.to_excel("X_app1.xlsx",index=True)
Y_app.to_csv("Y_app1.csv",index=True)
data_app=pd.concat([X_app,Y_app], axis=1)
data_app.to_csv("data_app.csv",index=True)
from joblib import dump
dump(forest , 'RandomForestClassifier.joblib')
