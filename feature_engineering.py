import seaborn as sb
import pandas as pd
import sklearn
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    global df
    #path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_1_psy_devcen_email_cen_devavgtime_pagerank.csv'
    #path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_2_psy_devcen_email_cen_devavgtime_pagerank_eigen.csv'
    path = '/Users/apple4u/Desktop/goksel tez/findings_data/2_3_psy_devcen_email_cen_devavgtime_pagerank_eigen_emailharmonic.csv'

    df = pd.read_csv(path)

    df.drop(columns='employee_name', inplace=True)
    df.drop(columns='user_id', inplace=True)
    df.drop(columns='device_degree_centrality', inplace=True)
    df.drop(columns='device_closeness_centrality', inplace=True)
    df.drop(columns='device_betweenness_centrality', inplace=True)
    df.drop(columns='email_degree_centrality', inplace=True)
    df.drop(columns='email_closeness_centrality', inplace=True)
    df.drop(columns='email_betweenness_centrality', inplace=True)

    #df.drop(columns='insider_label', inplace=True) # for kendall correlations

    df.insider_label.fillna(0, inplace=True)
    df.device_avg_duration.fillna(0, inplace=True)
    df.email_eigenvector.fillna(method="pad", inplace=True)
    df.email_harmonic_centrality.fillna(method="pad", limit=1, inplace=True)

    print(df.info)
    X = df.iloc[:, :11].values
    #X = df.iloc[:, [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]].values # selcting teh columns one by one Openness is 0
    y = df.insider_label.values

    # splitting raw dataset to three: train validation test split and printing sizes
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=9, stratify=y)
    # to keep insider/non insider ratio for splitted dataset
    print('=================  data size =========================')
    print('x (raw) size:', len(X), 'y (raw) size:', len(y))
    print('xTrainInitial size:', len(xTrain), 'yTrainInitial size:', len(yTrain))
    xTrain, xVldtn, yTrain, yVldtn = train_test_split(xTrain, yTrain, test_size=0.4, random_state=8, stratify=yTrain)
    print('xTrain size:', len(xTrain), 'yTrainSize:', len(yTrain))
    print('xVal size:', len(xVldtn), 'yValSize:', len(yVldtn))
    print('xTest size:', len(xTest), 'yTest size:', len(yTest))

    reSampler = RandomOverSampler(random_state=0)
    xTrainOverSampled, yTrainOverSampled = reSampler.fit_sample(xTrain, yTrain)
    print('ReSampled xTrain size:', len(xTrainOverSampled), 'Resampled yTrainSize:', len(yTrainOverSampled))

    corr = df.corr(method='kendall')
    rcParams['figure.figsize'] = 14.7, 8.27
    sb.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,
           cmap="YlGnBu",
          annot=True)
    #plt.show()
    #plt.savefig('kendal_with_pagerank')

    #fig = sb.pairplot(df)
    #fig.savefig('pairplot')

    import scipy.stats  as stats
    # print(stats.pearsonr(df['device_betweenness_centrality'], df['device_degree_centrality']))
    # print(stats.pearsonr(df['device_betweenness_centrality'], df['device_closeness_centrality']))
    #print(stats.pearsonr(df['device_betweenness_centrality'], df['device_avg_duration']))
    #print(stats.pearsonr(df['insider_label'], df['device_avg_duration']))
    # print(stats.pearsonr(df['email_betweenness_centrality'], df['email_degree_centrality']))
    # print(stats.pearsonr(df['email_betweenness_centrality'], df['email_closeness_centrality']))
    # print(stats.pearsonr(df['email_degree_centrality'], df['email_pagerank']))
    # print(stats.pearsonr(df['email_betweenness_centrality'], df['email_pagerank']))
    # print(stats.pearsonr(df['device_eigenvector'], df['device_betweenness_centrality']))
    print(stats.pearsonr(df['email_harmonic_centrality'], df['email_eigenvector']))
    print(stats.pearsonr(df['email_harmonic_centrality'], df['email_pagerank']))
    print(stats.pearsonr(df['email_eigenvector'], df['email_pagerank']))


    # these are temporrary getting ff values
    classifier = RandomForestClassifier(random_state=8, n_estimators=100, max_features='auto', n_jobs=1, class_weight={0:1,1:1})
    #classifier = AdaBoostClassifier(classifier)
    # classifier = MLPClassifier()
    #classifier = SVC(kernel="sigmoid", degree=1) #degree has not changed the accuracy
    # classifier = SVC(kernel="poly")
    # classifier = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0)) #increases f1 and accuracy
    #classifier = DecisionTreeClassifier(max_depth=5) #7 is optimum

    classifier.fit(xTrainOverSampled, yTrainOverSampled)
    predictions = classifier.predict(xVldtn)
    print('Classification report:]\n', classification_report(yVldtn, predictions))

    c = confusion_matrix(yVldtn, predictions)
    print('Confusion matrix:\n', c)
    TN, FP, FN, TP = c[0][0], c[0][1], c[1][0], c[1][1]
    print ("TN:" + str(TN))
    print ("FP:" + str(FP))
    print ("FN:" + str(FN))
    print ("TP:" + str(TP))

    #  [True Negative False positive] [ False Negative True Positive ]

    print(sklearn.metrics.accuracy_score(yVldtn, predictions, normalize=True, sample_weight=None))
    print(sklearn.metrics.matthews_corrcoef(yVldtn, predictions, sample_weight=None))

    probabilities = classifier.predict_proba(X)[:, 1]
    #print(probabilities)

