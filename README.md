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
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("bmi.csv")
df.head()
```

![Screenshot 2025-04-29 173040](https://github.com/user-attachments/assets/f859678d-211d-4de9-acc1-bde494845fc9)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![Screenshot 2025-04-29 173539](https://github.com/user-attachments/assets/f309ed5c-27c4-4879-94e8-51c62709dd38)

```
df.dropna()
```

![Screenshot 2025-04-29 173848](https://github.com/user-attachments/assets/a3b1f035-c70f-41b4-b104-1bb4637c7ce4)

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

![Screenshot 2025-04-29 173920](https://github.com/user-attachments/assets/36786c76-e960-441d-afc5-c65a85977b06)

```
from sklearn.preprocessing import StandardScaler
```
```
df1=pd.read_csv("bmi.csv")
df1.head()
```

![Screenshot 2025-04-29 174015](https://github.com/user-attachments/assets/c8ee8563-7281-4f6f-b8b6-53213db7d675)

```
sc=StandardScaler()
```
```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![Screenshot 2025-04-29 174104](https://github.com/user-attachments/assets/0147307e-3bc2-400c-92bd-c92c83f7a3ed)

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
```
```
df3=pd.read_csv("bmi.csv")
df3.head()
```

![Screenshot 2025-04-29 174330](https://github.com/user-attachments/assets/d3424d0a-4699-4418-b22d-ddd71830549f)

```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

![Screenshot 2025-04-29 174417](https://github.com/user-attachments/assets/a3845555-6a6d-4d44-a303-bc299fbf3f39)

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
```
```
df4=pd.read_csv("bmi.csv")
df4.head()
```

![Screenshot 2025-04-29 175502](https://github.com/user-attachments/assets/44b3e1ae-ba86-4f1f-b23f-afb4c96da576)

```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![Screenshot 2025-04-29 175945](https://github.com/user-attachments/assets/0c2bce4c-10ea-49e1-b970-22407349a6c2)

```
import pandas as pd
```
```
df=pd.read_csv("income(1) (1) (1).csv")
df.info()
```

![Screenshot 2025-04-29 180109](https://github.com/user-attachments/assets/b255667d-0475-47e3-a368-9e4a4199f880)

```
df
```

![Screenshot 2025-04-29 180214](https://github.com/user-attachments/assets/14bbca33-20f2-465f-89a0-64e9cdcd9022)

```
df.info()
```

![Screenshot 2025-04-29 180445](https://github.com/user-attachments/assets/8b76c4d8-af46-4ace-a958-3f9c718e9eaf)

```
df_null_sum=df.isnull().sum()
df_null_sum
```

![Screenshot 2025-04-29 180629](https://github.com/user-attachments/assets/2a316620-ab8b-473c-ab08-7c2ee9fc81f5)

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 180916](https://github.com/user-attachments/assets/2d66e769-018e-454d-af9a-b3b69f373889)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 181205](https://github.com/user-attachments/assets/09c68c3c-19e1-4272-b97b-cefe18b2c8c3)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
```
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```
```
rf.fit(X_train, y_train)
```

![Screenshot 2025-04-29 181407](https://github.com/user-attachments/assets/c5491bd7-13e9-4433-a59c-cdfecde2f0f6)

```
y_pred = rf.predict(X_test)
```
```
from sklearn.metrics import accuracy_score
```
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

![Screenshot 2025-04-29 181539](https://github.com/user-attachments/assets/8ca86014-6e89-4b97-accc-565c6793af54)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
```
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 182333](https://github.com/user-attachments/assets/a1580c39-9c94-45b9-87e5-e60733e600b1)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 182845](https://github.com/user-attachments/assets/ecaddb50-8b48-4b27-ab17-b11db5825339)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_chi2 = 6
```
```
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
```
```
selected_features_chi2 = X.columns[selector_chi2.get_support()]
```
```
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

![Screenshot 2025-04-29 183039](https://github.com/user-attachments/assets/4f84f5e8-26a4-48b4-8391-16e00653cdf9)

```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
```
```
X = df[selected_features]
y = df['SalStat']
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
```
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```
```
rf.fit(X_train, y_train)
```

![Screenshot 2025-04-29 183417](https://github.com/user-attachments/assets/42fafdac-08e3-435f-8bee-23f6369458ef)

```
y_pred = rf.predict(X_test)
```
```
from sklearn.metrics import accuracy_score
```
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

![Screenshot 2025-04-29 183713](https://github.com/user-attachments/assets/5052f92c-866e-4178-98f4-ba058ae141fe)

```
!pip install skfeature-chappers
```

![Screenshot 2025-04-29 183934](https://github.com/user-attachments/assets/8c117c42-0c56-4371-b71b-7f731ca660e5)

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 184146](https://github.com/user-attachments/assets/557d8e4a-058b-4ead-a224-3d98a0210d60)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 184248](https://github.com/user-attachments/assets/3b7d09d8-9ed2-4789-b48c-b6eb02751584)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

![Screenshot 2025-04-29 184457](https://github.com/user-attachments/assets/e17cd591-13af-4712-baf4-6c590ae4bd5f)

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![Screenshot 2025-04-29 184720](https://github.com/user-attachments/assets/c43f6db1-bde4-4a3a-8394-68f34598420b)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![Screenshot 2025-04-29 185114](https://github.com/user-attachments/assets/132f6d5d-f603-4548-b5ed-b4d4dd788492)


### RESULT:
Thus, the given data was successfully read and preprocessed.
