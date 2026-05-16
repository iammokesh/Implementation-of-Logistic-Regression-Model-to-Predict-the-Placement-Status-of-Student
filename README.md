# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Read the dataset and create a copy of the data.
2.Remove unwanted columns and check missing/duplicate values.
3.Convert categorical data into numerical form using Label Encoding.
4.Split the dataset into training and testing data and train the Logistic Regression model.
5.Predict the results and evaluate the model using accuracy, confusion matrix, and classification report.

## Program:
```python 
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:MOKESH C
RegisterNumber:212225240088
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")
print(data.head())

data1 = data.copy()

data1.drop(['sl_no', 'salary'], axis=1, inplace=True)

print("\nMissing values:\n", data1.isnull().sum())
print("\nDuplicate values:", data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])

x = data1.iloc[:, :-1]
y = data1['status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print("\nAccuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", confusion)

from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn import metrics

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Not Placed', 'Placed']
)

cm_display.plot()
plt.show()
```

## Output:
<img width="893" height="690" alt="Screenshot 2026-05-15 162003" src="https://github.com/user-attachments/assets/d992a6fd-4fb2-472e-bb54-97246bd03d0e" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
