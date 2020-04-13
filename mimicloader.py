import random

import pandas as pd
import numpy as np


class MIMICLoader:
    def load(self, path='', train_size=.8, kfolds=5, rand_seed=42, reduced=False):
        '''
        Loads the MIMIC csv at the path. Returns a tuple of:
            dataframe
            array of indicies for k-fold training data location
            array of indicies for k-fold testing data location
            array of indicies for validation data location

        Note, the validation data is indexed BEFORE doing K-Folds, so there is
        no overlap of the validation indexes with k-folds.

        :param path: Path to mimic csv
        :param train_size: percent of data to put in the training / validation
        :param kfolds: Number of folds
        :param rand_seed: Shuffle seed
        :param reduced: If true, uses only 10% of the data, useful for testing
        :return:
        '''
        data = pd.read_csv(path)
        rows = len(data.index)

        if (reduced):
            rows = int(rows * .1)
            data = data[:rows]

        indicies = np.arange(start=0, stop=len(data.index), step=1)
        random.seed(rand_seed)
        random.shuffle(indicies)

        train_length = int(len(indicies) * train_size)
        train_index = indicies[:train_length]
        test_index = indicies[train_length:]

        validation_set = data.iloc[test_index]

        train_set = []
        test_set = []

        fold_length = int(train_length / kfolds)
        for i in range(kfolds):
            train_index_np = np.array(train_index)
            test = train_index_np[i * fold_length:(i + 1) * fold_length]
            train = np.delete(train_index_np, train_index_np[i * fold_length:(i + 1) * fold_length])
            train_set.append(train)
            test_set.append(test)

        return data, train_set, test_set, validation_set