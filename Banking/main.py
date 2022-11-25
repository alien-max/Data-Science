import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os

path = os.path.dirname(os.path.abspath(__file__))
file_name = 'banking.txt'
file = os.path.join(path, file_name)
data = pd.read_csv(file)
data = data.dropna()

data['education'] = np.where(data['education'] == 'basic.4y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education'] == 'basic.9y', 'Basic', data['education'])

cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for var in cat_vars:
    cat_list = 'var_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1

cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]

cols = ['previous', 'euribor3m', 'job_blue-collar', 'job_retired', 'job_services', 'job_student', 'default_no', 'contact_cellular', 'month_apr', 'month_aug', 'month_dec', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'day_of_week_mon', 'poutcome_failure', 'poutcome_success']
x = data_final[cols]
y = data_final['y']

final_cm = 0

for i in range(10):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    final_cm += cm
    prfs = precision_recall_fscore_support(y_test, y_pred)

    n = i + 1
    print(str(n) + 'th evaluating ....\n')
    print('Confusion matrix: \n', cm)
    print('Precision: \n', prfs[0])
    print('Recall: \n', prfs[1])
    print('F-Score: \n', prfs[2])
    print('Support: \n', prfs[3])
    print('-------------------------------------------------------')

print('Final Confusion Matrix: \n', final_cm/10)