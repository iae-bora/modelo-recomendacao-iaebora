from numpy.core.fromnumeric import mean
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
from sklearn.metrics import silhouette_score
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def Treinar(locais, dataset, rodada, entrada, sample):
    #Separação de treino e teste
    #dataset.to_csv('C:\\Users\\Kaique\\OneDrive\\Área de Trabalho\\TCC\\dados1.csv')
    y = dataset['destino'].str.strip()
    x = dataset.drop(columns=['destino'])
    #y_sample = sample['destino'].str.strip()
    #x_sample = sample.drop(columns=['destino'])

    #Clusterização
    kmeans = KMeans(n_clusters=3)
    #grupos = kmeans.fit_predict(x)
    #x['Cluster'] = kmeans.labels_

    data_array = x.values
    #kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
    x["clusters"] = kmeans.fit_predict(data_array)
    #plot_kmeans = x.groupby("clusters").aggregate("mean").plot.bar()
    entrada.append(kmeans.predict([entrada]))


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)


    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)


    results = cross_validate(modelo, x, y, cv = 3, n_jobs = 3)
    previsoes_SVC = modelo.predict(X_train)
    acuracia = accuracy_score(y_train, previsoes_SVC) * 100
    acuracia_cross = mean(results['test_score'])*100
    
    #previsoes_sample = modelo.predict(x_sample)
    #acuracia_sample = accuracy_score(y_sample, previsoes_sample) * 100

    print("---------------------------------------------------------------------------------------------")
    print("Rodada: " + str(rodada))
    print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(X_train), len(X_test)))
    print("A acurácia foi de %.2f%%" % acuracia)
    print("A acurácia cross foi de %.2f%%" % acuracia_cross)
    #print(dataset['destino'].value_counts)
    saida = modelo.predict([entrada])

    return saida, acuracia, ""

def Recomendar(dataset, resposta):
    #Clusterização

    y = dataset['destino'].str.strip()
    x = dataset.drop(columns=['destino'])

    kmeans = KMeans(n_clusters=3)
    x['Cluster'] = kmeans.fit_predict(x)

    

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)
    previsoes_SVC = modelo.predict(X_train)
    acuracia = accuracy_score(y_train, previsoes_SVC) * 100
    print("A acurácia foi de %.2f%%" % acuracia)
    #print(x['destino'].value_counts)

    return modelo

