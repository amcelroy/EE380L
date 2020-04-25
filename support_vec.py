
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

ml = MIMICLoader()
data = ml.load("PythonCode/")

data, truth = ml.getDataSet(data, one_hot=False)

train, test, val = ml.train_test_split(data=data)

def sum(array):
    sum = 0
    for element in array:
        sum += element
    return sum

def avg(array):
    sum = 0
    for element in array:
        sum += element
    return sum/len(array)

train_accuracies = []
test_accuracies = []

true_positives = []
true_negatives = []
false_positives = []
false_negatives = []

sigmoid = False

for i in range(len(train)):
    print("Fitting {}:".format(i))

    train_data = data.iloc[train[i]].values
    truth_data = truth[train[i]]
    if(sigmoid):
        clf = svm.SVC(kernel="sigmoid")
    else:
        clf = svm.SVC(gamma="auto")
    clf.fit(train_data, truth_data)
    train_accuracy = clf.score(train_data, truth_data)
    test_accuracy = clf.score(data.iloc[test[i]].values, truth[test[i]])
    print("Train accuracy = {}".format(train_accuracy))
    print("Test accuracy = {}".format(test_accuracy))
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    prediction = clf.predict(data.iloc[test[i]].values)
    tn, fp, fn, tp = confusion_matrix(truth[test[i]], prediction).ravel()
    true_negatives.append(tn)
    false_positives.append(fp)
    false_negatives.append(fn)
    true_positives.append(tp)
    print('\n')

avg_train_accuracy = avg(train_accuracies)
avg_test_accuracy = avg(test_accuracies)

tp = sum(true_positives)
tn = sum(true_negatives)
fp = sum(false_positives)
fn = sum(false_negatives)

precision = tp / (tp+fp)
recall = tp / (tp + fn)
f1 = 2*(precision*recall)/(precision + recall)

print("\n\nOverall results:")

print("\nAverage train accuracy: {}".format(avg_train_accuracy))
print("Average test accuracy: {}".format(avg_test_accuracy))

print("\ntp = {}".format(tp))
print("tn = {}".format(tn))
print("fp = {}".format(fp))
print("fn = {}".format(fn))

print("\nPrecision (tp/(tp+fp)) = {}".format(precision))
print("Recall (tp/(tp+fn)) = {}".format(recall))
print("F1 = {}".format(f1))
