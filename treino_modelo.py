import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
import preparacao
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def Recomendacao(locais, dataset, rodada, entrada):


    #Separação de treino e teste
    #dataset.to_csv('C:\\Users\\Kaique\\OneDrive\\Área de Trabalho\\TCC\\dados1.csv')
    y = dataset['destino'].str.strip()
    x = dataset.drop(columns=['destino'])


    #Clusterização
    modeloCluster = KMeans(n_clusters=3)
    grupos = modeloCluster.fit_predict(x)
    x['Cluster'] = modeloCluster.labels_
    entrada.append(modeloCluster.predict([entrada]))


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)
    previsoes_SVC = modelo.predict(X_train)
    acuracia = accuracy_score(y_train, previsoes_SVC) * 100
    print("---------------------------------------------------------------------------------------------")
    print("Rodada: " + str(rodada))
    print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(X_train), len(X_test)))
    print("A acurácia foi de %.2f%%" % acuracia)
    #print(dataset['destino'].value_counts)
    saida = modelo.predict([entrada])

    return saida, acuracia

