import networkx
import pandas as pd
import pylab
import sklearn
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SVMSMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, roc_curve, recall_score, roc_auc_score, accuracy_score, \
    precision_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


classifier = RandomForestClassifier(random_state=0)

if __name__ == '__main__':
    global df
    path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_psy_devcen_email_cen_devavgtime.csv'
    #path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_1_psy_devcen_email_cen_devavgtime_pagerank.csv'

    df = pd.read_csv(path)
    df.drop(columns='employee_name', inplace=True)
    df.drop(columns='user_id', inplace=True)
    df.insider_label.fillna(0, inplace=True)
    df.device_avg_duration.fillna(0, inplace=True)

   # print(df.info)
    print(df.head())
    print(df.info)
    X = df.iloc[:, :12].values
    #X = df.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]].values # selcting teh columns one by one Openness is 0
    y = df.insider_label.values

    # splitting raw dataset to three: train validation test split and printing sizes
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)

    # to keep insider/non insider ratio for splitted dataset
    print('=================  data size =========================')
    print('x (raw) size:', len(X), 'y (raw) size:', len(y))
    print('xTrainInitial size:', len(xTrain), 'yTrainInitial size:', len(yTrain))
    xTrain, xVldtn, yTrain, yVldtn = train_test_split(xTrain, yTrain, test_size=0.25, random_state=1, stratify=yTrain)
    print('xTrain size:', len(xTrain), 'yTrainSize:', len(yTrain))
    print('xVal size:', len(xVldtn), 'yValSize:', len(yVldtn))
    print('xTest size:', len(xTest), 'yTest size:', len(yTest))

    # classifier = RandomForestClassifier(random_state=0)
    # classifier.fit(xTrain, yTrain)
    # predictions = classifier.predict(xVldtn)
    # print('Classification report:]\n', classification_report(yVldtn, predictions))
    # print('Confusion matrix:\n', confusion_matrix(yVldtn, predictions))
    # del classifier

    # KNN
    #classifier = KNeighborsClassifier(n_neighbors=7)
    # mat = classifier.kneighbors_graph(X, 2, mode='connectivity')
    # G = networkx.from_scipy_sparse_matrix(mat, create_using=networkx.Graph())
    # use networkx.draw to plot this G

    #SVM
    #classifier = SVC(kernel="poly")
    #classifier = SVC(kernel="sigmoid", degree=7) #degree has not changed the accuracy

    # Gaussian Process Classifier
    #classifier = GaussianProcessClassifier()
    #classifier = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0)) #increases f1 and accuracy

    #Decision Trees
    #classifier = DecisionTreeClassifier(max_depth=9) # 5 is optimum

    #Random Forest Classifier
    classifier = RandomForestClassifier(random_state=19)
    #classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) not tested

    # Multilayer Perceptron Classifier
    #classifier = MLPClassifier()

    #Adaptive Boosting
    #classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
    #classifier = AdaBoostClassifier(RandomForestClassifier(random_state=0))
    # ADABOOST changes f1 score for each run

    #Gaussian Naive Bayes
    #classifier = GaussianNB()

    #Linear and QuadraticDiscriminantAnalysis
    #classifier = QuadraticDiscriminantAnalysis()
    #classifier = LinearDiscriminantAnalysis()

    # oversampling insiders
    reSampler = RandomOverSampler(random_state=0)
    #reSampler = ADASYN(random_state=7)
    #reSampler = SVMSMOTE(random_state=39)
    xTrainOverSampled, yTrainOverSampled = reSampler.fit_sample(xTrain, yTrain)
    print('ReSampled xTrain size:', len(xTrainOverSampled), 'Resampled yTrainSize:', len(yTrainOverSampled))
    classifier.fit(xTrainOverSampled, yTrainOverSampled)
    predictions = classifier.predict(xVldtn)
    print('Classification report:]\n', classification_report(yVldtn, predictions))

    print('Confusion matrix:\n', confusion_matrix(yVldtn, predictions))
    #  [True Negative False positive] [ False Negative True Positive ]

    print(sklearn.metrics.accuracy_score(yVldtn, predictions, normalize=True, sample_weight=None))
    print(sklearn.metrics.matthews_corrcoef(yVldtn, predictions, sample_weight=None))

    probabilities = classifier.predict_proba(X)[:, 1]


    #Plotting probabilities
    # import matplotlib.pyplot as plt
    # import numpy as np
    # plt.figure(figsize=(8, 5))
    # plt.yscale('linear')
    # plt.title('Histogram of Probabilities')
    # plt.ylabel('Frequency of probabilities in linear scale')
    # plt.xlabel('Probability')
    # plt.hist(probabilities, bins=130)
    # plt.show()


    # PLOTTING ROC
    # fpr, tpr, thresholds = roc_curve(y, probabilities)  # Get the ROC Curve
    # print("AUC:" + str(roc_auc_score(y, probabilities)))
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 5))
    # # Plot ROC curve
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate (FPR)') #= 1 - Specificity Score'
    # plt.ylabel('True Positive Rate (TPR) ') #  = Recall Score
    # plt.title('Receiver Operation Characteristics (ROC)')
    # plt.show()

