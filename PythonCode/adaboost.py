class StumpLearner(object):
    def __init__(self):
        self.direction = 1
        self.node_feat_index = None
        self.thresh = None
        self.weight = None

class AdaBoost(object):

    def __init__(self, T):
        self.T = T

    def fit(self, X, y,random_selection=5):
        n_observations, n_feats = X.shape
        D = 1/n_observations*np.ones(n_observations)
        self.random_selection = random_selection
        self.classifier = []

        for _ in range(self.T):
            classifier = StumpLearner()
            min_err = float('inf')

            # pick random subset
            feats = np.random.randint(0,n_feats,self.random_selection)

            for feat_i in feats:
                thresholds = np.unique(X[:,feat_i])

                # bin thresholds, make threshold matrix from beginning?
                for threshold in thresholds:
                    direction = 1
                    prediction = np.ones(len(y))
                    prediction[X[:,feat_i] < threshold] = -1
                    #error = sum(D[y != prediction])
                    error = speedy_sum(D,y,prediction)

                    if error > 1/2:
                        error = 1-error
                        direction = -1
                    if error<min_err:
                        classifier.direction = direction
                        classifier.thresh = threshold
                        classifier.node_feat_index = feat_i
                        min_err = error

            classifier.weight = np.float(1/2)*np.log(1/error - 1)
            prediction = np.ones(len(y))
            prediction[(classifier.direction * X[:, classifier.node_feat_index] < classifier.direction * classifier.thresh)] = -1
            D = D*np.exp(-classifier.weight * y * prediction)
            D = D/np.sum(D)
            self.classifier.append(classifier)

    def predict(self,X):
        n_observations = X.shape[0]
        y_hat = np.zeros((n_observations,1))
        for WL in self.classifier:
            prediction = np.ones(np.shape(y_hat))
            prediction[(WL.direction * X[:, WL.node_feat_index] < WL.direction * WL.thresh)] = -1
            y_hat += WL.weight*prediction

        return np.sign(y_hat).flatten()

@jit
def speedy_sum(D,p,y):
    return np.sum(D[y != p])




T_star = [(i+1)*100 for i in range(10)]

a = []
for t in T_star:
    model = AdaBoost(T=t)
    model.fit(X_tr,y_tr)
    y_hat = model.predict(X_te)
    a.append(sum(y_hat == y_te)/len(y_te))
