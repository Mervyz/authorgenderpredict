# -*- coding: utf-8 -*-


import numpy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import Optimizasyon.BAT as bat
import Optimizasyon.FFA as ffa
import Optimizasyon.GWO as gwo
import Optimizasyon.MFO as mfo
import Optimizasyon.MVO as mvo
import Optimizasyon.PSO as pso
import Optimizasyon.WOA as woa
import Optimizasyon.fitnessFUNs as fitnessFUNs


def selectorDetail(algo, func_details, popSize, Iter, DataFile, DatasetSplitRatio=0.34):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]

    data_set = numpy.loadtxt(open(DataFile, "rb"), delimiter=",", skiprows=0)
    numRowsData = numpy.shape(data_set)[0]  # number of instances in the  dataset
    numFeaturesData = numpy.shape(data_set)[1] - 1  # number of features in the  dataset

    dataInput = data_set[0:numRowsData, 0:-1]
    dataTarget = data_set[0:numRowsData, -1]
    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget,
                                                                      test_size=DatasetSplitRatio, random_state=1)
    #

    #    numRowsTrain=numpy.shape(trainInput)[0]    # number of instances in the train dataset
    #    numFeaturesTrain=numpy.shape(trainInput)[1]-1 #number of features in the train dataset
    #
    #    numRowsTest=numpy.shape(testInput)[0]    # number of instances in the test dataset
    #    numFeaturesTest=numpy.shape(testInput)[1]-1 #number of features in the test dataset
    #

    dim = numFeaturesData

    if algo == 0:
        print("önce Pso mu yoksa fn1 mi çalışacak")
        x = pso.PSO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 1:
        x = mvo.MVO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 2:
        x = gwo.GWO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 3:
        x = mfo.MFO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 4:
        x = woa.WOA(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 5:
        x = ffa.FFA(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 6:
        x = bat.BAT(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)

    # Evaluate MLP classification model based on the training set
    #    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
    #   x.trainAcc=trainClassification_results[0]
    #  x.trainTP=trainClassification_results[1]
    # x.trainFN=trainClassification_results[2]
    # x.trainFP=trainClassification_results[3]
    # x.trainTN=trainClassification_results[4]

    # Evaluate MLP classification model based on the testing set
    # testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)

    reducedfeatures = []
    for index in range(0, dim):
        if x.bestIndividual[index] == 1:
            reducedfeatures.append(index)
    reduced_data_train_global = trainInput[:, reducedfeatures]
    reduced_data_test_global = testInput[:, reducedfeatures]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(reduced_data_train_global, trainOutput)

    # Compute the accuracy of the prediction

    target_pred_train = knn.predict(reduced_data_train_global)
    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc = acc_train

    target_pred_test = knn.predict(reduced_data_test_global)
    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc = acc_test

    # print('Test set accuracy: %.2f %%' % (acc * 100))

    # x.testTP=testClassification_results[1]
    # x.testFN=testClassification_results[2]
    # x.testFP=testClassification_results[3]
    # x.testTN=testClassification_results[4]

    return x


def selector(algo, func_details, popSize, Iter, completeData):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]

    DatasetSplitRatio = 0.34  # Training 66%, Testing 34%

    print("completeData")
    print(completeData)
    DataFile = "..//test_datasets//" + completeData

    data_set = numpy.loadtxt(open(DataFile, "rb"), delimiter=",", skiprows=0)
    numRowsData = numpy.shape(data_set)[0]  # number of instances in the  dataset
    numFeaturesData = numpy.shape(data_set)[1] - 1  # number of features in the  dataset

    dataInput = data_set[0:numRowsData, 0:-1]
    dataTarget = data_set[0:numRowsData, -1]
    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget,
                                                                      test_size=DatasetSplitRatio, random_state=1)
    #

    #    numRowsTrain=numpy.shape(trainInput)[0]    # number of instances in the train dataset
    #    numFeaturesTrain=numpy.shape(trainInput)[1]-1 #number of features in the train dataset
    #
    #    numRowsTest=numpy.shape(testInput)[0]    # number of instances in the test dataset
    #    numFeaturesTest=numpy.shape(testInput)[1]-1 #number of features in the test dataset
    # 

    dim = numFeaturesData

    if algo == 0:
        x = pso.PSO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 1:
        x = mvo.MVO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 2:
        x = gwo.GWO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 3:
        x = mfo.MFO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 4:
        x = woa.WOA(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 5:
        x = ffa.FFA(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)
    if algo == 6:
        x = bat.BAT(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)

    # Evaluate MLP classification model based on the training set
    #    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
    #   x.trainAcc=trainClassification_results[0]
    #  x.trainTP=trainClassification_results[1]
    # x.trainFN=trainClassification_results[2]
    # x.trainFP=trainClassification_results[3]
    # x.trainTN=trainClassification_results[4]

    # Evaluate MLP classification model based on the testing set   
    # testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)

    reducedfeatures = []
    for index in range(0, dim):
        if x.bestIndividual[index] == 1:
            reducedfeatures.append(index)
    print("reduced features", reducedfeatures)
    reduced_data_train_global = trainInput[:, reducedfeatures]
    reduced_data_test_global = testInput[:, reducedfeatures]
    print("reduced reduced_data_train_global", reduced_data_train_global)
    if(function_name=="FN1"):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(reduced_data_train_global, trainOutput)
        target_pred_train = knn.predict(reduced_data_train_global)
        target_pred_test = knn.predict(reduced_data_test_global)
    elif(function_name=="FN2"):
        clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0)
        clf.fit(reduced_data_train_global, trainOutput)
        target_pred_train = clf.predict(reduced_data_train_global)
        target_pred_test = clf.predict(reduced_data_test_global)
    elif(function_name=="FN3"):
        clf =  RandomForestClassifier(min_samples_split=2, n_estimators=1000, min_samples_leaf=10)
        clf.fit(reduced_data_train_global, trainOutput)
        target_pred_train = clf.predict(reduced_data_train_global)
        target_pred_test = clf.predict(reduced_data_test_global)


    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc = acc_train


    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc = acc_test

    print('SelectorFonk Gradient Boosting Test set accuracy: %.2f %%' % (acc_test * 100))

    #x.testTP=testClassification_results[1]
    # x.testFN=testClassification_results[2]
    # x.testFP=testClassification_results[3]
    # x.testTN=testClassification_results[4] 

    return x,reducedfeatures

#####################################################################
