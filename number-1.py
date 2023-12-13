# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:04.925766Z","iopub.execute_input":"2023-12-13T14:15:04.926136Z","iopub.status.idle":"2023-12-13T14:15:04.963030Z","shell.execute_reply.started":"2023-12-13T14:15:04.926108Z","shell.execute_reply":"2023-12-13T14:15:04.961607Z"}}

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass
import warnings

warnings.filterwarnings('ignore')





# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:04.965442Z","iopub.execute_input":"2023-12-13T14:15:04.965776Z","iopub.status.idle":"2023-12-13T14:15:04.979593Z","shell.execute_reply.started":"2023-12-13T14:15:04.965748Z","shell.execute_reply":"2023-12-13T14:15:04.978167Z"}}
#reading the data set 
data = '/kaggle/input/traffic-prediction-dataset/Traffic.csv'

df = pd.read_csv(data, header=None)

#deletes first row, as that is header
df.drop(labels=0,inplace=True)


#naming columns
col_names = ['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount','TruckCount', 'Total', 'Traffic Situation']

df.columns = col_names


# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:04.983237Z","iopub.execute_input":"2023-12-13T14:15:04.983612Z","iopub.status.idle":"2023-12-13T14:15:04.997504Z","shell.execute_reply.started":"2023-12-13T14:15:04.983581Z","shell.execute_reply":"2023-12-13T14:15:04.995899Z"}}
#selcting target value and removing categories for accuracy
X = df.drop(['Traffic Situation'], axis=1)

y = df['Traffic Situation']

# spliting training and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)


# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:05.000151Z","iopub.execute_input":"2023-12-13T14:15:05.001154Z","iopub.status.idle":"2023-12-13T14:15:05.051845Z","shell.execute_reply.started":"2023-12-13T14:15:05.001115Z","shell.execute_reply":"2023-12-13T14:15:05.050717Z"}}
import category_encoders as ce


# encode categorical variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['Time','Day of the week', 'Date','CarCount', 'BikeCount', 'BusCount','TruckCount','Total'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:05.052947Z","iopub.execute_input":"2023-12-13T14:15:05.053261Z","iopub.status.idle":"2023-12-13T14:15:05.057606Z","shell.execute_reply.started":"2023-12-13T14:15:05.053230Z","shell.execute_reply":"2023-12-13T14:15:05.056899Z"}}
# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier

# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:05.058623Z","iopub.execute_input":"2023-12-13T14:15:05.058887Z","iopub.status.idle":"2023-12-13T14:15:08.704696Z","shell.execute_reply.started":"2023-12-13T14:15:05.058864Z","shell.execute_reply":"2023-12-13T14:15:08.703440Z"}}
# instantiate the classifier with n_estimators = 1000

rfc_1000 = RandomForestClassifier(n_estimators=1000, random_state=0)



# fit the model to the training set

rfc_1000.fit(X_train, y_train)



# Predict on the test set results

y_pred_1000 = rfc_1000.predict(X_test)



# Check accuracy score 
from sklearn.metrics import accuracy_score

print('Model accuracy score with 1000 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_1000)))

# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:08.706514Z","iopub.execute_input":"2023-12-13T14:15:08.706886Z","iopub.status.idle":"2023-12-13T14:15:08.766534Z","shell.execute_reply.started":"2023-12-13T14:15:08.706855Z","shell.execute_reply":"2023-12-13T14:15:08.764545Z"}}
feature_scores = pd.Series(rfc_1000.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores

# %% [code] {"execution":{"iopub.status.busy":"2023-12-13T14:15:08.768554Z","iopub.execute_input":"2023-12-13T14:15:08.769833Z","iopub.status.idle":"2023-12-13T14:15:08.810964Z","shell.execute_reply.started":"2023-12-13T14:15:08.769796Z","shell.execute_reply":"2023-12-13T14:15:08.809825Z"}}
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_1000))