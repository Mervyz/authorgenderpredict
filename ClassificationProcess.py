import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


class SiniflandirmaAlg():

    def __init__(self, data_set, testsize=0.3):
        # sütun ve satır sayısı bilgileri okunur
        numRowsData = np.shape(data_set)[0]  # number of instances in the  dataset
        numFeaturesData = np.shape(data_set)[1] - 1  # number of features in the  dataset
        print(numRowsData)
        print("  ")
        print(numFeaturesData)

        # veride sınıf bilgisi y değişkeninde diğer tüm bilgiler x değerinde tutulur
        X = data_set.iloc[0:numRowsData, 0:-1]
        y = data_set.iloc[0:numRowsData, -1]

        # train ve test olarak samples ayrılır
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=testsize)

    def KNN(self, n_neighbors=3, acc=True, classificationReport=True, ConfusionMatrix=True, meanAbsulateError=True,
                              meansquaredError=True, rootMeanSquaredError=True):
        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors)
        print("scaler dan önce")

        scaler = preprocessing.StandardScaler().fit(self.X_train)

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        print("fit fomk. önce")
        knn.fit(self.X_train, self.y_train)
        print("predict fonk. önce")
        y_pred = knn.predict(self.X_test)
        print("acc değerlerinden önce")
        self.BasariDegerleriGoster(y_pred,acc,classificationReport,ConfusionMatrix,meanAbsulateError,meansquaredError,rootMeanSquaredError)

    def BasariDegerleriGoster(self, y_pred, acc=True, classificationReport=True, confusionMatrix=True, meanAbsulateError=True,
                              meansquaredError=True, rootMeanSquaredError=True):
        if acc:
            print("Accuracy Score:",accuracy_score(self.y_test, y_pred))
        if classificationReport:
            print('Classification Report:', classification_report(self.y_test, y_pred))
        if confusionMatrix:
            print('Confusion Matrix:',confusion_matrix(self.y_test, y_pred))
        if meanAbsulateError:
            print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, y_pred))
        if meansquaredError:
            print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, y_pred))
        if rootMeanSquaredError:
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test, y_pred)))

    def RandomForestRegressor(self,rf_n_estimators=20, rf_random_state=0, acc=False, classificationReport=True, confusionMatrix=True, meanAbsulateError=True,
                              meansquaredError=True, rootMeanSquaredError=True):
        regressor = RandomForestRegressor(n_estimators=rf_n_estimators,random_state=rf_random_state)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        self.BasariDegerleriGoster(y_pred,acc,classificationReport,confusionMatrix,meanAbsulateError,meansquaredError,rootMeanSquaredError)

    def RandomForest(self,rf_min_samples_split=2, rf_n_estimators=1000, rf_min_samples_leaf=10, acc=True, classificationReport=True, confusionMatrix=True, meanAbsulateError=True,
                              meansquaredError=True, rootMeanSquaredError=True):
        regressor =  RandomForestClassifier(min_samples_split=rf_min_samples_split, n_estimators=rf_n_estimators, min_samples_leaf=rf_min_samples_leaf)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        self.BasariDegerleriGoster(y_pred,acc,classificationReport,confusionMatrix,meanAbsulateError,meansquaredError,rootMeanSquaredError)

    def GradientBoosting(self, gb_n_estimators=20, gb_random_state=0, gb_learning_rate=1.0, gb_max_depth=1, acc=True, classificationReport=True,
                         confusionMatrix=True, meanAbsulateError=True, meansquaredError=True, rootMeanSquaredError=True):
        clf = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, max_depth=gb_max_depth, random_state=gb_random_state)
        clf.fit(self.X_train, self.y_train)
        y_pred=clf.predict(self.X_test)
        self.BasariDegerleriGoster(y_pred, acc, classificationReport, confusionMatrix, meanAbsulateError,
                                   meansquaredError, rootMeanSquaredError)