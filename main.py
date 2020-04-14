from mimicloader import MIMICLoader

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')
kfolds_train, kfolds_test, validation = \
    ml.train_test_split(data, train_size=.8, kfolds=5, rand_seed=42, reduced=True)

living = ml.getLiving(data)
dead = ml.getDeceased(data)

x = 0

