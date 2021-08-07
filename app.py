from flask import Flask, request
from flask_cors import CORS
import pickle, os

from numpy.core.records import record
import treino_modelo as treino

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return { 'message': 'Working correctly!' }

@app.route('/predict', methods=['POST'])
def predict():
    entrada = request.get_json(force=True)['answers']

    print("Teste entrada")
    lista = list(entrada.values())
    print(lista)

    locais = ['CINEMA', 'RESTAURANTE', 'SHOPPING', 'PARQUE', 'SHOW', 'MUSEU', 'BIBLIOTECA', 'ESTÁDIO', 'BIBLIOTECA', 'JOGOS', 'TEATRO', 'BAR']
    modelo = treino.Recomendacao(locais)

    recomendacao = modelo.predict([lista])
    return { 'recomendacao': recomendacao }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
