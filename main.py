from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')

data, truth = ml.getDataSet(data)

train, test, val = ml.train_test_split(data)

data = (data - data.mean())/data.std()

mimicnet = MIMICNet()
model = mimicnet.create(columns=data.shape[1])
mimicnet.compile()

epochs = 1000
hyper_param_knn = [3, 5, 7, 9, 11]

for j in range(len(train)):
    train_data = data.iloc[train[j]].values
    truth_data = truth[train[j]]
    knn = KNeighborsClassifier(n_neighbors=hyper_param_knn[j])
    knn.fit(train_data, truth_data)
    graph = knn.kneighbors_graph(data.iloc[test[j]].values)
    x = knn.score(data.iloc[test[j]].values, truth[test[j]])
    print('Error for knn={}: {}'.format(hyper_param_knn[j], x))
    # model.fit(train_data, truth_data,
    #           batch_size=512,
    #           epochs=epochs,
    #           verbose=2,
    #           shuffle=True,
    #           validation_data=(data[test[j]], truth[test[j]]))
    z = 0


x = 0

