from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ____________________________________________________________________________________
def FN_RF(I, trainInput, trainOutput, dim):
    data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput,
                                                                                                            trainOutput,
                                                                                                            test_size=0.34,
                                                                                                            random_state=1)

    reducedfeatures = []
    for index in range(0, dim):
        if I[index] == 1:
            reducedfeatures.append(index)

    reduced_data_train_internal = data_train_internal[:, reducedfeatures]
    reduced_data_test_internal = data_test_internal[:, reducedfeatures]

    rf = RandomForestRegressor(n_estimators=20, random_state=0)
    rf.fit(reduced_data_train_internal, target_train_internal)
    target_pred_internal = rf.predict(reduced_data_test_internal)
    mae = float(metrics.mean_absolute_error(target_test_internal, target_pred_internal))

    return mae


# _____________________________________________________________________

# ____________________________________________________________________________________
def FN1(I, trainInput, trainOutput, dim):
    data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput,
                                                                                                            trainOutput,
                                                                                                            test_size=0.34,
                                                                                                            random_state=1)

    reducedfeatures = []
    for index in range(0, dim):
        if I[index] == 1:
            reducedfeatures.append(index)

    reduced_data_train_internal = data_train_internal[:, reducedfeatures]
    reduced_data_test_internal = data_test_internal[:, reducedfeatures]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(reduced_data_train_internal, target_train_internal)
    target_pred_internal = knn.predict(reduced_data_test_internal)
    acc_train = float(accuracy_score(target_test_internal, target_pred_internal))

    fitness = 0.99 * (1 - acc_train) + 0.01 * sum(I) / dim

    print("dim: "+str(dim)+" I :"+str(I)+" Sum(I): "+str(sum(I)))
    print("Fitness Func acc", acc_train, " Fitness: ", fitness)
    return fitness


# _____________________________________________________________________
''' burada istersek FN1 yerine kendi fonksiyonuuzun adını yazabililiriz'''

def FN2(I, trainInput, trainOutput, dim):
    data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput,
                                                                                                            trainOutput,
                                                                                                            test_size=0.34,
                                                                                                            random_state=1)

    reducedfeatures = []
    for index in range(0, dim):
        if I[index] == 1:
            reducedfeatures.append(index)

    reduced_data_train_internal = data_train_internal[:, reducedfeatures]
    reduced_data_test_internal = data_test_internal[:, reducedfeatures]

    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(reduced_data_train_internal, target_train_internal)
    target_pred_internal = clf.predict(reduced_data_test_internal)

    acc_train = float(accuracy_score(target_test_internal, target_pred_internal))

    fitness = 0.99 * (1 - acc_train) + 0.01 * sum(I) / dim

    print("FN2 Fitness Func acc", acc_train, " Fitness: ", fitness)
    return fitness

def FN3(I, trainInput, trainOutput, dim):
    data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput,
                                                                                                            trainOutput,
                                                                                                            test_size=0.34,
                                                                                                            random_state=1)

    reducedfeatures = []
    for index in range(0, dim):
        if I[index] == 1:
            reducedfeatures.append(index)

    reduced_data_train_internal = data_train_internal[:, reducedfeatures]
    reduced_data_test_internal = data_test_internal[:, reducedfeatures]


    regressor =  RandomForestClassifier(min_samples_split=2, n_estimators=1000, min_samples_leaf=10)
    regressor.fit(reduced_data_train_internal, target_train_internal)
    target_pred_internal = regressor.predict(reduced_data_test_internal)



    acc_train = float(accuracy_score(target_test_internal, target_pred_internal))

    fitness = 0.99 * (1 - acc_train) + 0.01 * sum(I) / dim

    print("FN3 Fitness Func acc", acc_train, " Fitness: ", fitness)
    return fitness

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {0: ["FN1", -1, 1],
             1: ["FN2", -1, 1],
             2: ["FN3", -1, 1]

             }
    return param.get(a, "nothing")
