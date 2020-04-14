import random

import pandas as pd
import numpy as np


class MIMICLoader:

    def load(self, path=''):
        data = pd.read_csv(path)
        data = data.drop(columns=['age', 'patientweight', 'org_itemid', 'org_name', 'avg_num_drug_administered' \
                'max_drug_administered', 'total_input_drugs', 'tot_routes'])
        return data

    def getDeceased(self, data=pd.DataFrame):
        dead = data.loc[data['hospital_expire_flag'] == 1]
        return dead

    def getLiving(self, data=pd.DataFrame):
        dead = data.loc[data['hospital_expire_flag'] == 0]
        return dead

    def train_test_split(self, data=pd.DataFrame, train_size=.8, kfolds=5, rand_seed=42, reduced=False):
        '''
        Loads the MIMIC csv at the path. Returns a tuple of:
            array of indicies for k-fold training data location
            array of indicies for k-fold testing data location
            array of indicies for validation data location

        Note, the validation data is indexed BEFORE doing K-Folds, so there is
        no overlap of the validation indexes with k-folds.

        :param data: Pandas dataframe of the MIMIC data
        :param train_size: percent of data to put in the training / validation
        :param kfolds: Number of folds
        :param rand_seed: Shuffle seed
        :param reduced: If true, uses only 10% of the data, useful for testing
        :return:
        '''
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

        return train_set, test_set, validation_set
