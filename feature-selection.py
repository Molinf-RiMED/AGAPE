from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier as XGBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import svm
from tqdm import tqdm
import pandas as pd 
import os


#Parameters
scatter = True
bar = False
BEST_FEATURE = True
train = True

os.environ['XGBOOST_VERBOSE'] = '1'


df = pd.read_excel("data_complex.xlsx")

df["coded_label"] = df["LABEL"].apply(lambda x: 1 if x == "ACTIVE" else 0)

prova = df.drop(columns=["LABEL", "coded_label"])


x, y = prova.values, df["coded_label"].values


lista_model = [
    RandomForestClassifier(),
    svm.SVC(kernel='linear', C=1, random_state=17),
    svm.SVC(kernel='sigmoid', C=1, random_state=17),
    svm.SVC(kernel='rbf', C=1, random_state=17),
    XGBoostClassifier(),
    HistGradientBoostingClassifier(),
]

nome_modello = [
    "Random Forest",
    "SVM Linear",
    "SVM Sigmoid",
    "SVM RBF",
    "Hist-GBoost",
    "Hist Gradient Boosting"
]

for clf,nome_modello in tqdm(zip(lista_model, nome_modello), total=len(lista_model)):

    #! Sequential Forward Selection (sfs)
    sfs = SFS(clf, #! model to use for feature selection
            k_features=max_number_of_feature_in_CSV, #! number of features to select
            forward=True, #! technique to perform feature selection
            floating=False, #! required for the forward method
            scoring='f1', #! scoring metric to use (f1-score, precision, recall)
            cv=10, #! argument for k-fold cross-validation (0 to disable k-fold)
            verbose=2,
            n_jobs=-1 #! number of jobs to use (n_jobs=-1 to use all available cores)
            )
    
    sfs.fit(x, y)

    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev',figsize=(14, 10))
    plt.title('Sequential Forward Selection (w. StdErr)')
    plt.grid()
    plt.grid()
    plt.savefig(f"./feature_selection/{nome_modello}_sfs_v2.png")
    file = open(f"./feature_selection/{nome_modello}_sfs_v2.txt", "w")
    for i in sfs.subsets_.keys():
        file.writelines(str(sfs.subsets_[i]) + "\n")
    file.close()
