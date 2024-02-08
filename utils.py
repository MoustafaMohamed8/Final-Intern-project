## Main Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
import missingno
warnings.filterwarnings('ignore')
## sklearn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer

#import GridSearchCV
from sklearn.model_selection import GridSearchCV

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('Data.csv')
df.dropna(inplace=True)

df['LoanAmount_Category'] = pd.cut(df['LoanAmount'], bins=[0, 150, 300, float('inf')], labels=['Low', 'Mid', 'High'], right=False)
df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)
df['Dependents'].replace('3+',3,inplace=True)
df.drop('Loan_ID',axis=1,inplace=True)
num_cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Dependents']
categ_cols=['Gender','Married','Education','Self_Employed','LoanAmount_Category']
df['ApplicantIncome']=df['ApplicantIncome'].astype('float64')
df['LoanAmount_Category']=df['LoanAmount_Category'].astype('object')
df['Dependents']=df['Dependents'].astype('int64')
## to features and target
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']


## split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)


num_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(num_cols)),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

## Categorical

categ_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(categ_cols)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])


all_pipeline = FeatureUnion(transformer_list=[
                        ('num', num_pipline),
                        ('categ', categ_pipeline)
                    ])

## apply
_= all_pipeline.fit_transform(X_train)


def process_new(x_new):
    df_new=pd.DataFrame([x_new],columns=X_train.columns)
    

    ##Adjust the datatypes
    df_new['Gender']=df_new['Gender'].astype('object')
    df_new['Married']=df_new['Married'].astype('object')
    df_new['Dependents']=df_new['Dependents'].astype('int64')
    df_new['Education']=df_new['Education'].astype('object')
    df_new['Self_Employed']=df_new['Self_Employed'].astype('object')
    df_new['ApplicantIncome']=df_new['ApplicantIncome'].astype('float64')
    df_new['CoapplicantIncome']=df_new['CoapplicantIncome'].astype('float64')
    df_new['LoanAmount']=df_new['LoanAmount'].astype('float64')
    df_new['Loan_Amount_Term']=df_new['Loan_Amount_Term'].astype('float64')
    df_new['Credit_History']=df_new['Credit_History'].astype('float64')
    df_new['Property_Area']=df_new['Property_Area'].astype('object')
    df_new['LoanAmount_Category']=df_new['LoanAmount_Category'].astype('object')


    
   

    ## Apply the pipeline
    X_processed=all_pipeline.transform(df_new)


    return X_processed