from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')

data, truth = ml.getDataSet(data, one_hot=False)

train, test, val = ml.train_test_split(data=data)

for i in range(len(train)):
    print("Fitting {}:".format(i))
    train_data = data.iloc[train[i]].values
    truth_data = truth[train[i]]
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(train_data, truth_data)
    predicted = clf.predict(data.iloc[test[i]].values)
    print(confusion_matrix(truth[test[i]], predicted))
    print(classification_report(truth[test[i]], predicted))
    print('\n')

    train_error = clf.score(train_data, truth_data)
    test_error = clf.score(data.iloc[test[i]].values, truth[test[i]])
    print("Train error = {}".format(train_error))
    print("Test error = {}".format(test_error))
    print('\n\n\n')
