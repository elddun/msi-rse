import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):
    """
    Random subspace ensemble
    Komitet klasyfikatorow losowych podprzestrzeniach cech
    """

    def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=None):
        # Klasyfikator bazowy
        self.base_estimator = base_estimator
        # Liczba klasyfikatorow
        self.n_estimators = n_estimators
        # Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        # Tryb podejmowania decyzji
        self.hard_voting = hard_voting
        # Ustawianie ziarna losowosci
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        # Sprawdzenie czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # Przehowywanie nazw klas
        self.classes_ = np.unique(y)

        # Zapis liczby atrybutow
        self.n_features = X.shape[1]
        # Czy liczba cech w podprzestrzeni jest mniejsza od calkowitej liczby cech
        if self.n_subspace_features > self.n_features:
            raise ValueError(
                "Number of features in subspace higher than number of features.")
        # Wylosowanie podprzestrzeni cech
        self.subspaces = np.random.randint(
            0, self.n_features, (self.n_estimators, self.n_subspace_features))

        # Wyuczenie nowych modeli i stworzenie zespolu
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(
                clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

        return self

    def predict(self, X):
        # Sprawdzenie czy modele sa wyuczone
        check_is_fitted(self, "classes_")
        # Sprawdzenie poprawnosci danych
        X = check_array(X)
        # Sprawdzenie czy liczba cech się zgadza
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")
        if self.hard_voting:
            # Podejmowanie decyzji na podstawie twardego glosowania
            pred_ = []
            # Modele w zespole dokonuja predykcji
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
            # Zamiana na miacierz numpy (ndarray)
            pred_ = np.array(pred_)
            # Liczenie glosow
            prediction = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
            # Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]
        else:
            # Podejmowanie decyzji na podstawie wektorow wsparcia
            esm = self.ensemble_support_matrix(X)
            # Wyliczenie sredniej wartosci wsparcia
            average_support = np.mean(esm, axis=0)
            # Wskazanie etykiet
            prediction = np.argmax(average_support, axis=1)
            # Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        # Wyliczenie macierzy wsparcia
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)