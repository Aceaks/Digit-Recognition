import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

data_train_path='../input/digit-recognizer/train.csv'
data=pd.read_csv(data_train_path)
data.head()
data.shape
dataset=data.copy()
dataset.head()
y=dataset.pop('label')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,y,random_state=23)
X_train.shape
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
models=[KNeighborsClassifier(n_neighbors=7),RandomForestClassifier(),SVC()]
for model in models:
    model.fit(X_scaled,y_train)
    y_pred=model.predict(X_test_scaled)
    print(accuracy_score(y_pred,y_test))

accuracy_score(y_pred,y_test)
data_test_path='../input/digit-recognizer/test.csv'
testset=pd.read_csv(data_test_path)
testset=scaler.transform(testset)
test=np.array(testset)
a=test[0]
c=a.reshape((1,len(a)))
c.shape
c=scaler.transform(c)
b=c.reshape((28,-1))
b.shape
plt.imshow(b)
model.predict(c)
output=model.predict(testset)
ImageId=np.array([i for i in range(1,len(output)+1)])
submission=pd.DataFrame({'ImageId':ImageId,'Label':output})
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission_path='/kaggle/working/submission.csv'
pd.read_csv(submission_path)
model
joblib.dump(model, 'svm_digit_recognizer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

