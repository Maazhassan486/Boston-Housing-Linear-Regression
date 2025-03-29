import pandas as pd

df=pd.read_csv(r"C:\Users\maazg\Downloads\BostonHousing.csv");
# print(df.head(10));
y=df['medv'];
# print(y);
x=df.drop('medv',axis=1);
# print(x);

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=100);
# print(xtrain)
# print(ytrain)

import joblib
import os
from sklearn.linear_model import LinearRegression
model=r"C:\Users\maazg\PycharmProjects\PythonProject\.venv\model2.pkl";
if(os.path.exists(model)):
    lr=joblib.load(model);
    print("model loaded");
else:
     lr=LinearRegression();
     lr.fit(xtrain,ytrain);
     joblib.dump(lr,model);
     print("model trained");
y_lr_train_pred=lr.predict(xtrain);
# print(y_lr_train_pred);
y_lr_test_pred=lr.predict(xtest);
# print(y_lr_test_pred);

from sklearn.metrics import mean_squared_error,r2_score
y_lr_train_pred_mse=mean_squared_error(ytrain,y_lr_train_pred);
y_lr_train_pred_r2=r2_score(ytrain,y_lr_train_pred);

y_lr_test_pred_mse=mean_squared_error(ytest,y_lr_test_pred);
y_lr_test_pred_r2=r2_score(ytest,y_lr_test_pred);

lr_results=pd.DataFrame(["Linear Regression",y_lr_train_pred_mse,y_lr_train_pred_r2,y_lr_test_pred_mse,y_lr_test_pred_r2]).transpose();
lr_results.columns=["method","mse_train","r2_train","mse_test","r2_test"]
print(lr_results);

import matplotlib.pyplot as plt
plt.scatter(x=ytrain,y=y_lr_train_pred,alpha=0.3);
plt.plot();
plt.show();
