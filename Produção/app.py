from flask import Flask, request
from flask_cors import CORS
import pickle, os

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return { 'message': 'Working correctly!' }

@app.route('/predict', methods=['POST'])
def predict():
    entrada = request.get_json(force=True)['answers']

    modelo = open('modelo_DecisionTreeClassifier_64','rb')
    modelo = pickle.load(modelo)
    kmeans = open('modelo_kmeans_64','rb')
    kmeans = pickle.load(kmeans)

    cluster = kmeans.predict([entrada])[0]
    entrada.append(cluster)
    retorno = modelo.predict([entrada])[0]
    return { 'country': retorno }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)