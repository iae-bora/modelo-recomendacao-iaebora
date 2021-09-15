from flask import Flask, request
from flask_cors import CORS
import pickle, os
from numpy.core.records import record
from scipy.sparse import data
import treino_modelo as treino
import random
import pandas as pd
import numpy as np
from multiprocessing import  Pool
import numpy as np
import matplotlib.pyplot as plt
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

rodadas = []
acuracias = []
acuracias_sample = []
retornos = []
locais = ['CINEMA', 'RESTAURANTE', 'SHOPPING', 'PARQUE', 'SHOW', 'MUSEU', 'BIBLIOTECA', 'ESTÁDIO', 'BIBLIOTECA', 'JOGOS', 'TEATRO', 'BAR']


enum_saida = {
    "Parque" : 1,
    "Museu" : 2,
    "Cinema" : 3,
    "Shopping" : 4,
    "Bar" : 5,
    "Show" : 7,
    "Biblioteca" : 8,
    "Estádio" : 9,
    "Jogos" : 10,
    "Teatro" : 11,
}

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return { 'message': 'Working correctly!' }

@app.route('/predict', methods=['POST'])
def predict():
    entrada = request.get_json(force=True)['answers']

    dataset = pd.read_csv("dados.csv")
    dataset = dataset.drop(columns='Unnamed: 0')
    dataset = dataset.query("destino != 'OUTROS'")

    lista = list(entrada.values())
    print(lista)

    for i in range(entrada['qtd_destinos']):
        dataset = dataset[dataset['destino'].str.upper().isin(locais)]
        modelo = treino.Recomendar(dataset, entrada)
        recomendacao = modelo.predict([lista])[0]
        recomendacao_convertida = enum_saida[recomendacao]
        retornos.append(recomendacao_convertida)

        if recomendacao.upper() != 'OUTROS':
            locais.remove(recomendacao.upper())

    return retornos


@app.route('/Train', methods=['POST'])
def Train():
    dataset = pd.read_csv("dados_treino.csv")
    dataset = dataset.drop(columns='Unnamed: 0')
    dataset = dataset.query("destino != 'OUTROS'")
    sample = dataset.sample(n=50)
    print(dataset.shape)
    print(dataset)
    dataset = dataset[~dataset.isin(sample)]
    dataset = dataset.dropna()

    plt.ion()
    for rodada in range(20000):
        
        entrada = [random.randint(0,6), random.randint(0,5), random.randint(0,7), random.randint(0,4), random.randint(0,4), random.randint(0,5), random.randint(0,1), random.randint(15,60)]
        resposta, acuracia, acuracia_sample = treino.Treinar(locais, dataset, rodada, entrada, sample)
        acuracias.append(acuracia)

        rodadas.append(rodada)
        entrada.append(str(resposta[0]))

        dict_entrada = {
            'genero_musical' : entrada[0],
            'comida_favorita' : entrada[1],
            'filme_favorito' : entrada[2],
            'esporte_favorito' : entrada[3],
            'time' : entrada[4],
            'religiao' : entrada[5],
            'tem_filhos' : entrada[6],
            'idade' : entrada[7],
            'destino' : entrada[9]
        }


        dataset = dataset.append(dict_entrada, ignore_index=True)

        if rodada == 19999:
            plt.plot(rodadas, acuracias)
            plt.xlabel("Rodada de treino")
            plt.ylabel("Acurácia")
            plt.draw()
            plt.pause(0.05)
            plt.show()
    dataset.to_csv("dados.csv")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)