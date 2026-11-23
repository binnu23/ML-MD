#STEP 1 Import libraries

import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from skleran.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClssifier
from sklearn.essemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import(
  f1_score,
accuracy_score,
classification_report,
confusion_matrix
)

#STEP 2 : Load dataset

data=load_breast_cancer()

X=pd.DataFrame(
  data.data,
  columns=data feature_names
)

y=pd.series(data.target,name="target")

print("Dataset shape:",X.shape)
print("/nFirst 5 rows:")
print(X.head())

print("\nTarget values:")
print(y.value_counts())

#STEP 3:Split dataset

X_train,X_test,y_train,y_test=train_tets_split(
  X,
  y,
  test_size=0.20,
  random_state=42,
  stratify=y
)

print("\nTraining samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])

#STEP 4 : Define classification algorithms

models={

   "Logistic Regression":LogisticRgression(
     max_iter=5000
   ),

  "KNN": KNeighborsClassifier(),

  "Decision Tree": DecisionTreeClassifier(
    random_state=42
  ),

  "Random Forest": RandomForestClassifir(
    random_state=42
  ),

  "SVM":SVC()
}


#step 5 : Define hyperparameter grids

param_grids={

  "Logistic Regresson":{
    "model__c":[0.1,1,10],
    "model__solver":["liblinear"]
  },

  "KNN":{
    "model__n_neighbors":[3,5,7,9],
    "model__weights":["uniform","distance"]
  },

  "Decision Tree":{
    "model__max_depth":[None,3,5,10],
    "model__min_samples_split":[2,5,10]
  },

  "Random Forest":{
    "model__n_estimators":[100,200],
    "model__max_depth":[None,5,10],
    "model__min_samples_split":[2,5]
  },

  "SVM":{
    "model__C":[0.1,1,10],
    "model__kernel":["linear","rbf"],
    "model__gamma":["scale","auto"]
  }
}

#STEP 6: 




