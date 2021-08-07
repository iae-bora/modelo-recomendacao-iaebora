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
from sklearn.ensemble import RandomForestClassifier


def Recomendacao(locais):
    dataset = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTovXh_UG_XTyZoRjvsWsqOdLWpfLTxTLTG0_tj57dPg1AXlOMuqOsnezGQv8_PAQPnku8S-Nf-6PS4/pub?output=csv")

    #Renomear colunas
    colunas = {
        'Qual gênero musical você mais gosta?' : 'genero_musical',
        'Qual o seu tipo de comida favorito?' : 'comida_favorita',
        'Qual o seu estilo de filme favorito' : 'filme_favorito',
        'Qual seu esporte favorito? ' : 'esporte_favorito',
        'Torce para algum time?' : 'time',
        'Possui alguma religião?' : 'religiao',
        'Tem filhos?' : 'tem_filhos',
        'Qual a sua data de nascimento? ' : 'data_nascimento',
        'Que tipo de lugar você mais gosta de ir?' : 'destino'
    }

    dataset = dataset.rename(columns = colunas)

    dataset = preparacao.PadronizarValores(dataset)

    dataset = preparacao.CalcularIdade(dataset)

    dataset = preparacao.SepararDestinos(dataset, locais)


    #Separação de treino e teste
    y = dataset['destino'].str.strip()
    x = dataset.drop(columns=['destino'])


    #Clusterização
    modelo = AgglomerativeClustering(n_clusters=3)
    grupos = modelo.fit_predict(x)
    x['Cluster'] = modelo.labels_

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    previsoes_SVC = modelo.predict(X_train)
    acuracia = accuracy_score(y_train, previsoes_SVC) * 100
    print("A acurácia foi de %.2f%%" % acuracia)
    return modelo

