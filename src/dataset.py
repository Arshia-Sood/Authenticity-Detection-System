import pandas as pd
import json 


df1=pd.read_json(r'C:\Users\Arshia Sood\Desktop\Authenticity Detection System\jsonFiles\All_Beauty_5.json',lines=True)
df2=pd.read_json(r"C:\Users\Arshia Sood\Desktop\Authenticity Detection System\jsonFiles\AMAZON_FASHION_5.json",lines=True)
df3=pd.read_json(r"C:\Users\Arshia Sood\Desktop\Authenticity Detection System\jsonFiles\Appliances_5.json",lines=True)
df4=pd.read_json(r"C:\Users\Arshia Sood\Desktop\Authenticity Detection System\jsonFiles\Software_5.json",lines=True)

print(df1.columns)
print(df2.columns)
print(df3.columns)
print(df4.columns)

merged_df=pd.concat([df1,df2,df3,df4],ignore_index=True)

merged_df.to_csv("data/raw/dataset.csv",index=False)