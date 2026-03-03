import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r"data\raw\dataset.csv")
print(data.head())
print(data.shape)
print(data.columns)

print(data.isnull().sum())
data.dropna(subset=["overall","reviewText"],inplace=True)

data.drop(columns=["reviewTime","style","reviewerName","summary","image"],inplace=True)
print(data.head())
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)

print(data.describe())
print(data.info())

#converting the datatypes
data['overall']=data['overall'].astype(float)
data['verified']=data['verified'].astype(int)
data['vote']=data["vote"].fillna(0)
data["vote"]=data["vote"].astype(str).str.replace(",","",regex=False)
data["vote"]=pd.to_numeric(data["vote"],errors="coerce").fillna(0).astype(int)

import os

output_dir=os.path.join("data","preprocessed")
os.makedirs(output_dir,exist_ok=True)
preprocessed_data=os.path.join(output_dir,"preprocessed_dataset.csv")
data.to_csv(preprocessed_data,index=False)