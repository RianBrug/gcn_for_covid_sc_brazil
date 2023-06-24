from asyncio import constants
from numpy import argmax, ndarray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import constants
import numpy as np
import pandas as pd

class Utils:

    def transform_categorical_to_one_hot_encoding(data):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

    def transform_one_hot_encoding_to_categorical(data):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

    def regional_str_to_encoded(r) -> ndarray:
        switcher={
            "SUL": [0., 0., 0., 0., 0., 0., 1.],
            "MEIO OESTE E SERRA CATARINENSE": [0., 0., 0., 0., 1., 0., 0.],
            "GRANDE FLORIANOPOLIS": [0., 0., 1., 0., 0., 0., 0.],
            "ALTO VALE DO ITAJAI": [1., 0., 0., 0., 0., 0., 0.],
            "FOZ DO RIO ITAJAI": [0., 1., 0., 0., 0., 0., 0.],
            "GRANDE OESTE": [0., 0., 0., 1., 0., 0., 0.],
            "PLANALTO NORTE E NORDESTE": [0., 0., 0., 0., 0., 1., 0.]
        }
        return switcher.get(r, "")

    def tipoteste_str_to_encoded(r) -> ndarray:
        switcher={
            "BIOLOGIA MOLECULAR (RT-PCR)": [0., 0., 0., 0.],
            "IMUNOLOGICO (TESTE RAPIDO)": [0., 0., 0., 0.],
            "IGNORADO": [0., 0., 1., 0.],
            "NAO SE APLICA": [1., 0., 0., 0.]
        }
        return switcher.get(r, "IGNORADO")
    
    # receive a 'regional' as input and return a list of its neighbors, using the constants.REGIONS_AND_NEIGHBORS_DICT, every value of the neighbors list should be encoded, using the Utils.regional_str_to_encoded function
    def get_encoded_neighbors_of_region(regional):
        neighbors = constants.REGIONS_AND_NEIGHBORS_DICT[regional]
        encoded_neighbors = []
        for neighbor in neighbors:
            encoded_neighbors.append(Utils.regional_str_to_encoded(neighbor))
        return encoded_neighbors

    def get_neighbors_of_region(regional):
        return constants.REGIONS_AND_NEIGHBORS_DICT[regional]
    
     # get populations from assets/populacao_residente_sc_por_macroregiao.csv, normalize and add to graphs
    def get_population_from_csv():
        population = pd.read_csv('/Users/rcvb/Documents/tcc_rian/code/assets/populacao_residente_sc_por_macroregiao.csv', sep=';')
        population = population.set_index('region')
        population = population.to_dict()['population']
        # normalize population
        max_population = max(population.values())
        for key in population:
            population[key] = population[key] / max_population
        
        return population
    
    # implement get_target_cases
    def get_target_cases(data, regional, date):
        regional_data = data[data['regional'] == regional]
        regional_data = regional_data[regional_data['data'] == date]
        return regional_data['casosAcumulado'].values[0]

    def MDAPE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def MAE(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred))) * 100

    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred))) * 100

    def MSE(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred)) * 100

    def R2(y_true, y_pred):
        r2 = 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))
        return r2