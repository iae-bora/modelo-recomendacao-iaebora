import pandas as pd
import numpy as np
from itertools import chain


genero_musical ={
    'ROCK': 0,
    'SERTANEJO' : 1,
    'FORRÓ': 2,
    'GOSPEL' : 3,
    'POP' : 4,
    'FUNK' : 5,
    'RAP' : 6
}

comida_favorita = {
    'CHURRASCO' : 0,
    'CASEIRA' : 1,
    'VEGETARIANA' : 2,
    'FAST FOOD' : 3,
    'JAPONESA' : 4,
    'ITALIANA' : 5
}

filme_favorito = {
    'DRAMA' : 0,
    'AÇÃO' : 1,
    'AVENTURA' : 2,
    'ROMANCE' : 3,
    'ANIMAÇÃO' : 4,
    'SUSPENSE' : 5,
    'TERROR' : 6,
    'COMÉDIA' : 7
}

esporte_favorito = {
    'FUTEBOL' : 0,
    'BASQUETE' : 1,
    'VOLEI' : 2,
    'TÊNIS' : 3,
    'LUTAS' : 4
}

time = {
    'PALMEIRAS' : 0,
    'CORINTHIANS' : 1,
    'SANTOS' : 2,
    'SÃO PAULO' : 3,
    'NENHUM' : 4
}

religiao = {
    'CRISTIANISMO' : 0,
    'JUDÁISMO' : 1,
    'HINDUÍSMO' : 2,
    'BUDISMO' : 3,
    'ESPIRITISMO' : 4,
    'NENHUMA' : 5
}

tem_filhos = {
    'SIM' : 1,
    'NÃO' : 0
}

#Substitui os valores pelo respectivo código
def PadronizarValores(dataset):
    dataset['genero_musical'] = pd.to_numeric(dataset['genero_musical'].str.upper().replace(genero_musical), errors='coerce').fillna(7)
    dataset['comida_favorita'] = pd.to_numeric(dataset['comida_favorita'].str.upper().replace(comida_favorita), errors='coerce').fillna(6)
    dataset['filme_favorito'] = pd.to_numeric(dataset['filme_favorito'].str.upper().replace(filme_favorito), errors='coerce').fillna(8)
    dataset['esporte_favorito'] = pd.to_numeric(dataset['esporte_favorito'].str.upper().replace(esporte_favorito), errors='coerce').fillna(7)
    dataset['time'] = pd.to_numeric(dataset['time'].str.upper().replace(time), errors='coerce').fillna(5)
    dataset['religiao'] = pd.to_numeric(dataset['religiao'].str.upper().replace(religiao), errors='coerce').fillna(6)
    dataset['tem_filhos'] = pd.to_numeric(dataset['tem_filhos'].str.upper().replace(tem_filhos), errors='coerce').fillna(3)
    return dataset

#Calcular idade
def CalcularIdade(dataset ):
    dataset['data_nascimento'] = dataset['data_nascimento'].replace('09/12/1377', '09/12/2000')
    dataset['data_nascimento'] = dataset['data_nascimento'] + ' 00:00:00'
    dataset['data_nascimento'] = pd.to_datetime(dataset['data_nascimento'], format='%d/%m/%Y %H:%M:%S')
    dataset['idade'] = 2021 - dataset['data_nascimento'].dt.year
    for index, row in dataset.iterrows():
      if dataset['idade'][index] > 60 or dataset['idade'][index] < 10:
        dataset['idade'][index] = dataset['idade'].mean()
    return dataset

def SepararDestinos(dataset, locais):
    for index, row in dataset.iterrows():
        dataset['destino'][index] = row['destino'].replace('Jogos (kart, boliche, paintball...)', 'Jogos').strip()

    auxDataset = dataset['destino'].str.split(',').astype(np.object)

    lens = auxDataset.str.len()

    dataset = pd.DataFrame({
        'genero_musical' : dataset['genero_musical'].values.repeat(lens),
        'comida_favorita' : dataset['comida_favorita'].values.repeat(lens),
        'filme_favorito' : dataset['filme_favorito'].values.repeat(lens),
        'esporte_favorito' : dataset['esporte_favorito'].values.repeat(lens),
        'time' : dataset['time'].values.repeat(lens),
        'religiao' : dataset['religiao'].values.repeat(lens),
        'tem_filhos' : dataset['tem_filhos'].values.repeat(lens),
        'idade' : dataset['idade'].values.repeat(lens),
        'destino' : list(chain.from_iterable(auxDataset.tolist())), 
    })
    

    for index, row in dataset.iterrows():
        if dataset['destino'][index].strip().upper() not in locais:
            dataset['destino'][index] = 'OUTROS'

    return dataset
