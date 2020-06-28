import numpy as np
from rse import RandomSubspaceEnsemble

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mojerse import MyRandomSubspaceEnsemble
datasets = ['ionosphere', 'australian', 'breastcan', 'diabetes','ecoli4','german', 'glass4','cryotherapy','yeast6','sonar']
ransdom_st = 772
for data, dataset in enumerate(datasets):
    print("###########################################################################")
    print("dataset:", datasets[data])

    dataset = datasets[data]
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")

    X = dataset[:, :-1]

    y = dataset[:, -1].astype(int)

    print(X.shape,y.shape)

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    clfs = {
    'RSE': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=ransdom_st),    
    'mojeRse':MyRandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, feat_per_subsp=5, hard_voting=True, random_state=ransdom_st),    
    'AdaBoost': AdaBoostClassifier(base_estimator =DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, random_state=ransdom_st),
    'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, random_state=ransdom_st, bootstrap=True),

    # 'RSE_SM':RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=ransdom_st),
    # 'mojeRSE_SM':MyRandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, feat_per_subsp=5, hard_voting=False, random_state=ransdom_st),
    # 'AdaBoost': AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=ransdom_st),
    # 'Bagging': BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=ransdom_st, bootstrap=True),
    
    }

    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
    scores = np.zeros((len(clfs), n_splits * n_repeats))


    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
    print("ksztalt scores", scores.shape)
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)




    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


    np.save('results', scores)

    scores = np.load('results.npy')
    print("Folds:\n", scores)

    from scipy.stats import ttest_ind

    alfa = .05
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)


    from tabulate import tabulate

    headers = ["RSE", "mrse", "adaboost","bagging",]
    names_column = np.array([["RSE"], ["mrse"], ["adaboost"],["bagging"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)


    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)


    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)