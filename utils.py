import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing

def preprocess(df):
    print('---------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns[:-1]:
        missingRows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missingRows))

    cols = ['class']
    df = pd.get_dummies(df[:-1])

    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns = df.columns)
    df_scaled['class_Iris-setosa'] = df['class_Iris-setosa']
    df_scaled['class_Iris-versicolor'] = df['class_Iris-versicolor']
    df_scaled['class_Iris-virginica'] = df['class_Iris-virginica']
    df = df_scaled
    
    return df_scaled