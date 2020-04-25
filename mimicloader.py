import random

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MIMICLoader:
    def load(self, path=''):
        df = pd.read_csv(path+'mimic_dataset.csv')
        df.drop(columns=['Unnamed: 0'],inplace=True)
        df = df[df['patientweight'].notna()]
        fillcols = {'hospital_expire_flag':0,'age':df.age.mean(),'NumDrugs':0,'num_procedures':0,'curr_service':0,'num_serv':0,'num_transfers':0,'curr_careunit':0,\
                    'avg_los':df.avg_los.mean(),'tot_los':df.tot_los.mean(),'num_unique_reads':df.num_unique_reads.mean(),\
                   'total_reads':df.total_reads.mean(),'uinique_caregivers':df.uinique_caregivers.mean(),'total_icd9':df.total_icd9.mean(),'total_icu_hours':0,\
                   'avg_icu_hours':0,'total_icu_stays':0,'avg_num_drug_administered':0,'max_drug_administered':0,'total_input_drugs':0,'tot_routes':0,\
                   'tot_org':0,'org_name':0,'org_itemid':0}
        df.fillna(value=fillcols,inplace=True)
        serv = df.curr_service.unique()
        care = df.curr_careunit.unique()
        org = df.org_name.unique()

        def replace_stuff(s):
            new_dic = {}
            i = 1
            for j in s:
                if j != 0:
                    new_dic[j] = i
                    i +=1
            return new_dic
        df.replace({'curr_service':replace_stuff(serv),'curr_careunit':replace_stuff(care),'org_name':replace_stuff(org)},inplace=True)
        df = df.apply(np.int64)
        df = df.drop(columns=['subject_id','hadm_id','total_input_drugs','total_icu_hours','max_drug_administered','org_name'])
        return df

    def covariance(self, data=pd.DataFrame, plot=False):
        '''
        plot correlation's matrix to explore dependency between features
        '''
        # init figure size
        corr = data.corr()

        fig, ax = plt.subplots(1, 1, figsize=(15,20))
        sns.heatmap(corr, annot=True,
                    fmt=".1f",
                    square=True,
                    linewidths=.4,
                    cbar_kws={"shrink": .7},
                    vmax=1,
                    vmin=0,
                    center=0,
                    ax=ax)
        ax.set_ylim(corr.shape[0] - 1, -0.5)
        ax.figure.subplots_adjust(bottom=0.3)
        plt.show()
        fig.savefig('corr.png')
        return corr

    def getDeceased(self, data=pd.DataFrame, number=None):
        dead = data.loc[data['hospital_expire_flag'] == 1]
        dead = dead.drop(columns='hospital_expire_flag')

        if(number):
            indicies = np.arange(0, number)
            random.shuffle(indicies)
            dead = dead.iloc[indicies]

        return dead

    def getLiving(self, data=pd.DataFrame, number=None):
        living = data.loc[data['hospital_expire_flag'] == 0]
        living = living.drop(columns='hospital_expire_flag')

        if(number):
            indicies = np.arange(0, number)
            random.shuffle(indicies)
            living = living.iloc[indicies]

        return living

    def getDataSet(self, df=pd.DataFrame, number=None, one_hot=False, balance=True):
        if number:
            living = self.getLiving(df, number=number)
            dead = self.getDeceased(df, number=number)
        else:
            living = self.getLiving(df)
            dead = self.getDeceased(df)

        if balance:
            min_value = min(dead.shape[0], living.shape[0])
            random_index = np.arange(min_value)
            random.shuffle(random_index)
            dead = dead.iloc[random_index]
            living = living.iloc[random_index]


        data = [dead, living]
        data = pd.concat(data, ignore_index=True)

        living_truth = np.zeros(living.shape[0])
        dead_truth = np.ones(dead.shape[0])

        if one_hot:
            class1 = np.concatenate([dead_truth, living_truth])
            class1 = np.expand_dims(class1, axis=1)
            class2 = 1 - class1
            all_truth = np.concatenate([class1, class2], axis=1)
        else:
            all_truth = np.concatenate([dead_truth, living_truth + 2])

        return data, all_truth

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
        val_index = indicies[train_length:]

        train_set = []
        test_set = []

        fold_length = int(train_length / kfolds)
        for i in range(kfolds):
            train_index_np = np.array(train_index)
            test = train_index_np[i * fold_length:(i + 1) * fold_length]
            train = np.delete(train_index_np, train_index_np[i * fold_length:(i + 1) * fold_length])
            train_set.append(train)
            test_set.append(test)

        return train_set, test_set, val_index
