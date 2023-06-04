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

