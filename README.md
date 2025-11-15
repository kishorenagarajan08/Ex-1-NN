<H3>ENTER THE NAME:  KISHORE N </H3>
<H3>ENTER YOUR REGISTER.NO:212223230106</H3>
<H3>EX. NO.1</H3>
<H3>14-09-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['Churn_Modelling.csv']))

print(df.describe())
print(df.head())

print("\nMissing values in dataset:")
print(df.isnull().sum())

df.isna().sum()
df = df.dropna()
duplicates = df.duplicated().sum()
df = df.drop_duplicates()


df = df.drop(["RowNumber","CustomerId","Surname"], axis=1)


X = df.drop(["Exited","Geography","Gender"], axis=1)
y = df["Exited"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

print("Length of X_test ",len(X_test))
print("Length of X_train",len(X_train))

```


## OUTPUT:
### Describe:
<img width="1174" height="712" alt="image" src="https://github.com/user-attachments/assets/7d2f9ad7-5ee4-4a86-a4b2-4a921a87fd04" />

### Missing values:
<img width="469" height="385" alt="image" src="https://github.com/user-attachments/assets/7686e13d-2e57-41ed-a68f-9cbd11385921" />

### Normalised dataset:
<img width="873" height="460" alt="image" src="https://github.com/user-attachments/assets/adc90a3f-1de6-4813-9f9e-08a6a8b04f58" />

### Train test split:
<img width="1508" height="222" alt="image" src="https://github.com/user-attachments/assets/da9bd8aa-b7f1-4735-b267-f09a23246c51" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


