
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')

data, truth = ml.getDataSet(data, one_hot=False)

train, test, val = ml.train_test_split(data=data)

def array_avg(array):
    sum = 0
    for element in array:
        sum += element
    return sum

print(len(data))
print(len(truth))

predictions = []
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []

for i in range(len(train)):
    print("Fitting {}:".format(i))
    train_data = data.iloc[train[i]].values
    truth_data = truth[train[i]]
    print(len(train_data))
    print(len(truth_data))
    clf = svm.SVC(gamma="auto")
    clf.fit(train_data, truth_data)
    train_accuracy = clf.score(train_data, truth_data)
    test_accuracy = clf.score(data.iloc[test[i]].values, truth[test[i]])
    print("Train accuracy = {}".format(train_accuracy))
    print("Test accuracy = {}".format(test_accuracy))
    predictions.append(clf.predict(data.iloc[test[i]].values))
    tn, fp, fn, tp = confusion_matrix(truth[test[i]], predictions[i]).ravel()
    true_negatives.append(tn)
    false_positives.append(fp)
    false_negatives.append(fn)
    true_positives.append(tp)
    print('\n\n\n')

tp = int(array_avg(true_positives))
tn = int(array_avg(true_negatives))
fp = int(array_avg(false_positives))
fn = int(array_avg(false_negatives))

print("tp = {}".format(tp))
print("tn = {}".format(tn))
print("fp = {}".format(fp))
print("fn = {}".format(fn))
