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

ransdom_st = 772

#słownik z klasyfikatorami których będziemy używać w testach
clfs = {
    # 'RSE': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=ransdom_st),    
    # 'mojeRse':MyRandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, feat_per_subsp=5, hard_voting=True, random_state=ransdom_st),    
    # 'AdaBoost': AdaBoostClassifier(base_estimator =DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, random_state=ransdom_st),
    # 'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, random_state=ransdom_st, bootstrap=True),

    'RSE_SM':RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=ransdom_st),
    'mojeRSE_SM':MyRandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=ransdom_st), n_estimators=10, feat_per_subsp=5, hard_voting=False, random_state=ransdom_st),
    'AdaBoost': AdaBoostClassifier(base_estimator=GaussianNB(),n_estimators=10, random_state=ransdom_st),
    'Bagging': BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=ransdom_st, bootstrap=True),
    
}

#zestawy danych wybrane na potrzeby testów
datasets = ['ionosphere', 'australian', 'breastcan', 'diabetes','ecoli4','german', 'glass4','cryotherapy','yeast6','sonar']

#stratyfikowana wielokrotna walidacja krzyżowa 5-krotna z 2-oma powtórzeniamy
n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

#macierz gdzie liczba wierszy to liczba testowanych modeli, kolumny to zestawy danych, a 3. wymiar to wyniki uzyskane w proacesie walidacji 5*2
scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))
#print(scores)
# xxx = np.zeros((2,2,10))
# print(xxx)

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    #print("ksztalt datasetu", datasets[data_id], " wymiary" , X.shape)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            #print(X[train].shape[1])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)




#zapisujemy wyniki 
np.save('results', scores)
#wczytujemy wyniki
#print("Folds:\n", scores)
# mean = np.mean(scores, axis=2).T
# std = np.std(scores, axis=2).T
# for clf_id, clf_name in enumerate(clfs):
#     print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


scores = np.load('results.npy')
#print(scores)
print("\nScores:\n", scores.shape)
#Uśredniamy wyniki po foldach, axis 2 oznacza że uśredniony zostanie 3 wymiar, następnie transponujemy macierz żeby w kolumnach znajdowały się metody, a w wierszach zbiory danych
mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

mean = np.mean(mean_scores, axis=0)
print("\nFor all datasets:", mean)
#Rangi. Im wyższa tym metoda jest lepsza. 
from scipy.stats import rankdata
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

    
clfsname=[]
for a in clfs:
    clfsname.append(a)
clfsname = np.array(clfsname)
print("\nModels:", clfsname)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:", mean_ranks)


#statystyczne testy parowe, test Wilcoxona
##alfa =0.05 
from scipy.stats import ranksums

#teest wilcoxa
alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

from tabulate import tabulate



headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)


stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)



# print("####################################################################")
# print(scores.shape)

# for i,dataset in enumerate(datasets):
#      print("dataset:", datasets[i])
#     from scipy.stats import ttest_ind

#     alfa = .05
#     t_statistic = np.zeros((len(clfs), len(clfs)))
#     p_value = np.zeros((len(clfs), len(clfs)))

#     for i in range(len(clfs)):
#         for j in range(len(clfs)):
#             t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
#     print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)