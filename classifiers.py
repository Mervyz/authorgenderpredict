import time
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
import os
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

dict_classifiers = {
    #"Logistic Regression": LogisticRegression(C=1000.0, penalty='l1'),  # n_jobs=1, C=1e5 'penalty': 'l1', 'C': 1000.0
    "Nearest Neighbors": KNeighborsClassifier(),
    #"Linear SVM": SVC(),
    #"Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    #"Decision Tree": tree.DecisionTreeClassifier(),
    # "Random Forest": RandomForestClassifier(n_estimators=1000),
    #"Random Forest": RandomForestClassifier(min_samples_split=2, n_estimators=1000, min_samples_leaf=10),
    # 'min_samples_split': 2, 'n_estimators': 100, 'criterion': 'gini', 'min_samples_leaf': 10
    #"Neural Net": MLPClassifier(alpha = 1),
    # "Neural Net": MLPClassifier(solver='lbfgs', random_state=1, activation='logistic', alpha=1.0, hidden_layer_sizes=(15,)),
    # "Naive Bayes": GaussianNB(),
    # "AdaBoost": AdaBoostClassifier(),
    # "QDA": QuadraticDiscriminantAnalysis(),
    # "Gaussian Process": GaussianProcessClassifier()
}


def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.

    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
    So it is best to train them on a smaller dataset first and
    decide whether you want to comment them out or not based on the test accuracy score.
    """

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        #t_start = time.clock()
        classifier.fit(X_train, Y_train)
        #t_end = time.clock()

        t_diff = 0
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                        'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models


def label_encode(df, list_columns):
    """
    This method one-hot encodes all column, specified in list_columns

    """
    for col in list_columns:
        le = LabelEncoder()
        col_values_unique = list(df[col].unique())
        le_fitted = le.fit(col_values_unique)

        col_values = list(df[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        df[col] = col_values_transformed


def expand_columns(df, list_columns):
    for col in list_columns:
        colvalues = df[col].unique()
        for colvalue in colvalues:
            newcol_name = "{}_is_{}".format(col, colvalue)
            df.loc[df[col] == colvalue, newcol_name] = 1
            df.loc[df[col] != colvalue, newcol_name] = 0
    df.drop(list_columns, inplace=True, axis=1)


def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    X_train_post = df_train['post'].values
    X_test_post = df_test['post'].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test, X_train_post, X_test_post


def get_train_test_combination(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values

    return df_train, df_test, X_train, Y_train, X_test, Y_test


    pass


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                       columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_.sort_values(by=sort_by, ascending=False))
    return df_


def display_corr_with_col(df, col):
    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0, len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()

def anafonksiyon(data_set,testsize=0.3):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)
    return display_dict_models(batch_classify(X_train, y_train, X_test, y_test))
def main():
    # veriyi dosyadan okuma işlemi
    files = [csv for csv in os.listdir("ExperimentFile") if
             csv.endswith(".csv")]
    # Main loop for reading and writing files
    for file in files:
        print(file)
        name="ExperimentFile//" + file
        with open(name, "r") as inFile:
            data_set = pd.read_csv(name)

            # veri hakkındaki bilgieri yazdırma işlemi
            print(data_set.head())
            print(data_set.info())

            # sınıflandırma işlemi
            a = anafonksiyon(data_set, 0.3)
            print(name+ " --------------------------------------")





if __name__ == "__main__":
    main()
