# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('income.csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/dc63c989-7f96-4d67-a2ec-4d319b03b53c)
![image](https://github.com/user-attachments/assets/025a43a7-74a6-43f5-ae59-4a31cfcf0b29)
![image](https://github.com/user-attachments/assets/f4d4d24d-d880-4d32-a58c-c7fbab63bff2)
![image](https://github.com/user-attachments/assets/690802aa-0ffd-4bdd-9b97-37f8a6a88497)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,' greater than 50,000':1})
print(data['SalStat'])
```
![image](https://github.com/user-attachments/assets/8f5a94cb-618e-4d47-ac91-ba4bdaf66087)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/1cc913c1-d2ec-490b-a554-94aecd000393)
![image](https://github.com/user-attachments/assets/83984149-86d0-44c8-ad90-d5ea6f91fe6d)
![image](https://github.com/user-attachments/assets/9f9b5781-42b2-453c-98d9-43a6adb9c693)
![image](https://github.com/user-attachments/assets/11cd905f-ead6-43d8-990f-25bf98b12ceb)
![image](https://github.com/user-attachments/assets/ca3dba30-9a1c-46b4-b764-e59473c6b586)
# y=new_data['SalStat'].values
![image](https://github.com/user-attachments/assets/6ab90924-7567-4035-a8a5-a91528d9aae2)
# x=new_data[feature].values
![image](https://github.com/user-attachments/assets/fd4a29a6-8f64-42eb-bfab-5328ad79ea4c)
# train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
# KNN_classifier=KNeighborsClassifier(n_neighbors=5)
```
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/user-attachments/assets/1da5dace-6e7d-4d9c-aa2e-e48b60183aa6)
# prediction=KNN_classifier.predict(test_x)
# confusionMatrix= confusion_matrix(test_y,prediction)
![image](https://github.com/user-attachments/assets/872a185f-b236-44d9-b28d-c806a14f6319)
# accuracy_score=accuracy_score(test_y,prediction)
![image](https://github.com/user-attachments/assets/5f691892-cecd-48c1-8518-4f908db031a8)
# data.shape
![image](https://github.com/user-attachments/assets/3fcdac17-f95d-4f0b-b87c-c33d0b7b5ebb)

# RESULT:
Thus feature scaling and selection is performed.
