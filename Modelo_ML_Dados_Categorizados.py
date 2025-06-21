# Dados Categorizados: Modelage
# Projeto Completo de Marchine Learning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from zlib import crc32
import sklearn as sk
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
# Bibliotecas para baixar os dados e criar o diretorio
import os
import tarfile
import urllib
# Baixando os dados
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("C:/", "Users", "nonat", "OneDrive", "Desktop", "Instituto Inteligência de Dados",
                            "Ciencia de Dados", "DataScience", "Modelo_Regressão_Python_ML", "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "C:/Users/nonat/OneDrive/Desktop/Instituto Inteligência de Dados/Ciencia de Dados/DataScience/Modelo_Regressão_Python_ML/datasets/housing/housing.csv"
# Criar a função para baixar o arquivo, criar o diretorio e abrir o arquvio
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# Carregar os Dados
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
dados = load_housing_data()
# Gerandoos dados categoricos
dados_categoricos = dados[["ocean_proximity"]]
subset=dados_categoricos.iloc[5674:6123]
print(subset)
# Categorizar os dados, ou seja, ordenar ou qualificar as classes
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
dados_categoricos_enconde=ordinal_encoder.fit_transform(dados_categoricos)
print(dados_categoricos_enconde[:7])
print(ordinal_encoder.categories_)
# Categorizar os dados pelo OneHotEnconder
from sklearn.preprocessing import OneHotEncoder
cat_encoder= OneHotEncoder()
dados_categoricos_enconde_1hot=cat_encoder.fit_transform(dados_categoricos)
print(dados_categoricos_enconde_1hot)
# Modelo basico de transformadores para classes
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__ (self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X, y=None):
            return self
    def transform(self,X):
            rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
            population_per_househld = X[:,population_ix]/X[:,households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
                return np.c_[X,rooms_per_household,population_per_househld,
                             bedrooms_per_room]
            else:
                return np.c_[X,rooms_per_household,population_per_househld]
att_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = att_adder.fit_transform(dados.values)
print(housing_extra_attribs)