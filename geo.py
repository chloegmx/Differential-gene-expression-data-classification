#Exploratory data analysis libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Scikit_learn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#model evaluation libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomisedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay

#load data and pre-processing
df = pd.read_csv("GEO_dataset.csv")
df.head()
df.shape()
df.["target"].value_counts()
df["target"].value_counts().plot(kind="bar",color=["orange", "red"]);
df.info()
#check for missing values
df.isna.sum()
df.describe()
df.gender.value_counts()
#compare gender and target column
pd.crosstab(df.target,df.gender)
#plot of the comparision
pd.crosstab(df.target,df.gender).plot(kind="bar",
                                      figsize =(10,6),
                                      color = ["salmon", "lightblue"]);
plt.title("Alzheimers frequency according to gender")
plt.xlabel("0 = No Disease,1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"]);
#plot of age vs frequency to develop alzheimers
df.age.plot.hist();

#correlation matrix
df.corr()
corr_matrix = df.corr()
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot= True,
                 linewidth=0.5, fmt=".2f",
                 cmap="YlGnBu");
#test-train split
x = df.drop(columns=['Samples','target'])
y = df["target"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

#fitting data into the 5 models
models = {"Logistic Regression":LogisticRegression(),
          "Random Forest":RandomForestClassifier(),
          "Adaboost":AdaBoostClassifier(),
          "Support Vector Classifier":SVC(C = 10, gamma = 0.01),
          "Naive Bayes":GaussianNB()}
def fit_and_score(models,x_train,x_test,y_train,y_test):
  np.random.seed(32)
  model_scores= {}
  for  name,model in models.items():
    model.fit(x_train,y_train)
    model_scores[name]=model.score(x_test,y_test)
   return model_scores

model_scores = fit_and_scores(models = models,
                              x_train=x_train,
                              x_test=x_test,
                              y_train=y_train,
                              y_test=y_test)
model_scores

#comparision of results from the 5 models
model_compare = pd.DataFrame(model_scores,index=["accuracy"])
model_compare.T.plot.bar()
