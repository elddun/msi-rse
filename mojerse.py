import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MyRandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):


    def __init__(self, base_estimator=None, n_estimators=10, feat_per_subsp=5, hard_voting=True, random_state=None):
            # Klasyfikator bazowy
            self.base_estimator = base_estimator
            # Liczba klasyfikatorow
            self.n_estimators = n_estimators
            # Liczba cech w jednej podprzestrzeni
            self.feat_per_subsp = feat_per_subsp
            # Tryb podejmowania decyzji
            self.hard_voting = hard_voting
            # Ustawianie ziarna losowosci
            self.random_state = random_state
            np.random.seed(self.random_state)
            
    def prepare_subspace(self, X, feat_count, train=True):

        if train==True:
            self.cols_id=[]
            
            for i in range(self.n_estimators):
                self.cols_id.append(np.random.randint(0, feat_count, size=self.feat_per_subsp))

            self.cols_id = np.array(self.cols_id)
            # print("indeksy kolumn:\n",cols_id)

        subsp=[]
        for i in range(self.n_estimators):
            subsp.append(X[:,self.cols_id[i]])
        subsp = np.array(subsp)
        #print("ksztalt podprzestrzeni uczących:", subsp.shape)
        #print("podprzestrzenie uczące:\n",subsp)
        return (subsp)

    def fit(self, X,y):
        X,y = check_X_y(X,y)
        self.classes = np.unique(y)

        self.feat_count = X.shape[1]

        if self.feat_per_subsp > self.feat_count:
            raise ValueError("Ilośc atrybutów w podprzestrzeni jest większa niż liczba atrybutów")
        #losujemy indeksy kolumn
        train_subsp = self.prepare_subspace(X, self.feat_count, train=True)
        
        self.ensemble=[]
        for i in range(self.n_estimators):
           # print("wyuczamy model nr:",i,"na podprzestrzeni nr:", i)
            self.ensemble.append(clone(self.base_estimator).fit(train_subsp[i],y))

    def predict(self, X):
        check_is_fitted(self, "classes")
        X = check_array(X)
        if X.shape[1] != self.feat_count:
            raise ValueError("Ilość atrybutów się nie zgadza")
        test_subsp = self.prepare_subspace(X, self.feat_count, train=False)
       
        if self.hard_voting:            
            voting = []
            for i, ensemble_member in enumerate(self.ensemble):
                voting.append(ensemble_member.predict(test_subsp[i]))
            voting = np.array(voting)

            voting_counted = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 1, arr=voting.T)

            return self.classes[voting_counted]
        else:
            
            probabilities = []
            for i, ensemble_member in enumerate(self.ensemble):
                probabilities.append(ensemble_member.predict_proba(test_subsp[i]))
            probabilities = np.array(probabilities)
           # print("probabilities:\n", probabilities)
            average_support = np.mean(probabilities, axis = 0)
            prediction= np.argmax(average_support, axis = 1)
            return self.classes[prediction]
            
