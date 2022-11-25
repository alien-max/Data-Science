import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os

path = os.path.dirname(os.path.abspath(__file__))
file_name = 'diabetes.csv'
file = os.path.join(path, file_name)
data = pd.read_csv(file)
data = data.dropna()

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = data[cols]
y = data['Outcome']

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
