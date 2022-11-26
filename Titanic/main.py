import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os

path = os.path.dirname(os.path.abspath(__file__))
file_name = 'titanic.xlsx'
file = os.path.join(path, file_name)
data = pd.read_excel(file)
data = data.dropna()
cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare']
data = data[cols]

cat_list = 'var_Sex'
cat_list = pd.get_dummies(data['Sex'], prefix='Sex')
data1 = data.join(cat_list)
data = data1

cat_vars = ['Sex']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]

cols = ['Pclass', 'Age', 'Fare', 'Sex_female', 'Sex_male']
y = data_final['Survived']
x = data_final[cols]

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