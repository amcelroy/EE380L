from mimicloader import MIMICLoader

ml = MIMICLoader()
mimic_dataframe, kfolds_train, kfolds_test, validation = \
    ml.load('mimic_dataset.csv', train_size=.8, kfolds=5, rand_seed=42, reduced=True)

x = 0

