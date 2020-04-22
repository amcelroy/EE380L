from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')

data, truth = ml.getDataSet(data, balance=True)
#ml.covariance(data, plot=True)

train, test, val = ml.train_test_split(data)

#data = (data - data.mean())/data.std()

epochs = 1000
hyper_param_knn = [3, 5, 7, 9, 11]

for j in range(len(train)):
    train_data = data.iloc[train[j]].values
    test_data = data.iloc[test[j]].values

    # pca = PCA(n_components=2)
    # train_data = pca.fit_transform(train_data)
    # test_data = pca.fit_transform(test_data)

    knn = KNeighborsClassifier(n_neighbors=hyper_param_knn[j], algorithm='brute')
    knn.fit(train_data, truth[train[j]])
    graph = knn.kneighbors_graph(train_data)
    train_error = knn.score(train_data, truth[train[j]])
    test_error = knn.score(test_data, truth[test[j]])
    print('Error for knn={}: Training {}, Testing {}'.format(hyper_param_knn[j], train_error, test_error))
    # model.fit(train_data, truth_data,
    #           batch_size=512,
    #           epochs=epochs,
    #           verbose=2,
    #           shuffle=True,
    #           validation_data=(data[test[j]], truth[test[j]]))
    z = 0



x = 0
