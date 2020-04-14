from mimicloader import MIMICLoader

ml = MIMICLoader()
data = ml.load('mimic_dataset.csv')
kfolds_train, kfolds_test, validation = \
    ml.train_test_split(data, train_size=.8, kfolds=5, rand_seed=42, reduced=True)

living = ml.getLiving(data, number=60)
dead = ml.getDeceased(data, number=60)

train, test, val = ml.train_test_split(living)

x = 0

