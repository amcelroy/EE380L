from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, normalize, Normalizer

from MIMICNet import MIMICNet
from mimicloader import MIMICLoader
import numpy as np

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')

data, truth = ml.getDataSet(data, one_hot=True)

train, test, val = ml.train_test_split(data)

mimicnet = MIMICNet()
model = mimicnet.create(columns=data.shape[1])
mimicnet.compile()

epochs = 1000

for j in range(len(train)):
    train_data = data.iloc[train[j]].values
    test_data = data.iloc[test[j]].values

    train_data = normalize(train_data, axis=1)
    test_data = normalize(test_data, axis=1)

    # pca = PCA()
    # train_data = pca.fit_transform(train_data)
    # test_data = pca.fit_transform(test_data)

    model.fit(train_data, truth[train[j]],
              batch_size=64,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              callbacks=[mimicnet],
              validation_data=(test_data, truth[test[j]]))
    z = 0


x = 0
