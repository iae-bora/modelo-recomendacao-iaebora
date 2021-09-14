from numpy.core.fromnumeric import mean
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')


def Treinar(locais, dataset, rodada, entrada, sample):
    #Separação de treino e teste
    
    y = dataset['destino'].str.strip()
    x = dataset.drop(columns=['destino'])
    #y_sample = sample['destino'].str.strip()
    #x_sample = sample.drop(columns=['destino'])

    #Clusterização
    kmeans = KMeans(n_clusters=3)

    data_array = x.values
    x["clusters"] = kmeans.fit_predict(data_array)
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

    return modelo

