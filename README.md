# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries (numpy, pandas, matplotlib, sklearn).
2. Load the dataset using pandas.
3. Display dataset records and check for missing values.
4. Plot a scatter graph to view the relationship between study hours and marks.
5. Define independent (X) and dependent (Y) variables.
6. Split the dataset into training and test sets.
7. Import LinearRegression from sklearn and create a model.
8. Train the model using training data.
9. Predict marks for test data and compare with actual values.
10. Evaluate model performance using MAE, MSE, RMSE, and R² score, and visualise results.

## Program :
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: TAMIZHSELVAN B
RegisterNumber: 212223230225
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

x=df.iloc[:,:-1].values
x


y=df.iloc[:,1].values
y

 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

### Outputs:
# Preview datasets
![EX_2_OUTPUT_1](https://github.com/user-attachments/assets/abf51aaf-f29d-4794-a2e9-b638dbf0c764)


# X initialization
![EX_2_OUTPUT_2](https://github.com/user-attachments/assets/3a252f54-c098-4d6c-a496-ea16d2adb1db)


# Y initialization
![EX_2_OUTPUT_3](https://github.com/user-attachments/assets/ae4d1ec9-f271-4f7c-9c22-a65b8f895357)

# Y_Predict 
![EX_2_OUTPUT_4](https://github.com/user-attachments/assets/0ee57c9f-0409-4bf3-ac46-c9270084d450)


![EX_2_OUTPUT_5](https://github.com/user-attachments/assets/9a8ccba3-ee77-4356-b0d8-2557f22f1737)

# Training Set
![EX_2_OUTPUT_6](https://github.com/user-attachments/assets/aa944e6c-8e33-4956-af3c-1dbdc2771048)

# Test Set
![EX_2_OUTPUT_7](https://github.com/user-attachments/assets/8dd7bfe0-c4d9-4d27-b523-c1fe23498b54)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
