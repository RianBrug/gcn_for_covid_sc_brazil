from asyncio import constants
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import constants
import numpy


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

    def regional_str_to_encoded(r):
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

    def tipoteste_str_to_encoded(r):
        switcher={
            "BIOLOGIA MOLECULAR (RT-PCR)": [0., 0., 0., 0.],
            "IMUNOLOGICO (TESTE RAPIDO)": [0., 0., 0., 0.],
            "IGNORADO": [0., 0., 1., 0.],
            "NAO SE APLICA": [1., 0., 0., 0.]
        }
        return switcher.get(r, "IGNORADO")
    