# Projeto Completo de Marchine Learning
from sklearn.compose import ColumnTransformer  # para padronizar todas as colunas
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
print(dados)
# print(dados.info())
# print(dados.columns)
# # # Plotando o histograma das features
# dados.hist(bins=50, figsize=(20, 10))  # Histograma de todas as features
# # Histograma de uma feature especifica
# dados["median_house_value"].hist(bins=50, figsize=(20, 10))
# plt.title("Valor Mediano da Casa")
# plt.xlabel("Valor da Casa")
# plt.ylabel("Total de residências")
# plt.show()
# print(dados["ocean_proximity"].value_counts())
# print(dados.describe())  # Descritiva das Features
# # Criando o conjunto de Teste e Guardando o Mesmo
# # # np.random.seed(42)
# def divi_treino_teste(data, teste_prop):
#      indice_embaralhado = np.random.permutation(len(data))
#      teste_conjunto_tamanho = int(len(data)*teste_prop)
#      teste_indices = indice_embaralhado[:teste_conjunto_tamanho]
#      treino_indices = indice_embaralhado[teste_conjunto_tamanho:]
#      return data.iloc[treino_indices], data.iloc[teste_indices]
# # # # # Chamando a Função
# conjunto_treino, conjunto_teste = divi_treino_teste(dados, 0.2)
# print(conjunto_treino.info(), conjunto_teste.info())
# # # # # Fixando as instância para que a cada novo conjunto de
# # # # # dados, não seja modificado, ou seja, fica fixo
# def teste_tamanho_checa(identidade, teste_prop):
#     return crc32(np.int64(identidade).tobytes()) < teste_prop*2**32
# # # # # Dividindo o conjunto de treino/teste e marcando as instâncias novas
# def divi_treino_teste_id(data, teste_prop, id_coluna):
#     ids = data[id_coluna]
#     in_teste_tamanho = ids.apply(lambda id_: teste_tamanho_checa(id_, teste_prop))
#     return data.loc[~in_teste_tamanho], data.loc[in_teste_tamanho]
# # # # # Acrescentando a Coluna Index
# dados_indxados = dados.reset_index()
# print(dados_indxados.head(1))
# train_set, test_set = divi_treino_teste_id(dados_indxados, 0.2, 'index')
# print(train_set.head(2), test_set.head(2))
# # # # # Garantindo a fixação das instancias para testes, usando outro identificador
# # # dados_indxados["id"] = dados["longitude"]*1000 + dados["latitude"]
# # # conjunto_treino, conjunto_teste = divi_treino_teste_id(
# # #     dados_indxados, 0.2, 'index')
# # # # # Usando a função do stick learn para dividir o conjunto de dados: Treino/Tteste
# # conjunto_treino, conjunto_teste = train_test_split(
# #     dados, test_size=0.2, random_state=42)
# # print(conjunto_treino, conjunto_teste)
# # print(dados.columns)
# # # # Categoriczando a renda mediana
dados["Renda_Mediana"] = pd.cut(dados["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
# dados["Renda_Mediana"].hist()
# # # # Adicionar Rótulos
# plt.xlabel('Renda Categorizada')
# plt.ylabel('Total de Pessoas')
# plt.title('Valor da renda mediana (em dolares americanos)')
# plt.show()
# # # Agora iremos estratificar a amostra: gerando os k-folds
dados_divide = StratifiedShuffleSplit(n_splits=1,
                                      test_size=0.2, random_state=42)
for treino_indices, teste_indices in dados_divide.split(dados, dados["Renda_Mediana"]):
    #     #     # Sem aspas para poder acessar os indices completos
    estrato_treino_set = dados.loc[treino_indices]
#     # Sem aspas para poder acessar os indices completos
    estrato_teste_set = dados.loc[teste_indices]
print(estrato_teste_set["Renda_Mediana"].value_counts(
) / len(estrato_teste_set))
# # # Reomvendo a Renda Mediana para que os Dados Retornem ao status inicial
for set_ in (estrato_treino_set, estrato_teste_set):
    set_.drop("Renda_Mediana", axis=1, inplace=True)
# # # Visualização dos Dados
# dados.plot(kind="scatter", x="longitude", y="latitude")
# plt.title("Identificar a Concentração de Imóveis Pela Localização")
# # # Analisar o preços dos imóveis: População, Preço: preços altos e baixos
# dados.plot(kind="scatter", x="longitude", y="latitude",
#            alpha=0.4, s=dados["population"]/100, label="População",
#            figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"),
#            colorbar=True)
# plt.legend()
# # plt.show()
# # # Olhando as Correlações
matriz_cor = dados.select_dtypes(include=[np.number]).corr()
print(matriz_cor["median_house_value"].sort_values(ascending=False))
# # # Verificando os atributos significativos
dados.plot(kind="scatter", x="median_income", y="median_house_value",
           alpha=0.1)
# # # Gerando novos atributos: total de comodos por domicilios
# # # Total de dormitorios por comodos
# # # total de pessoas por familias
dados["quartos_por_domicilios"] = dados["total_rooms"]/dados["households"]
dados["comodos_por_domicilios"] = dados["total_bedrooms"]/dados["total_rooms"]
dados["pessoas_por_domicilios"] = dados["population"]/dados["households"]
matriz_cor = dados.select_dtypes(include=[np.number]).corr()
print(matriz_cor["median_house_value"].sort_values(ascending=False))
# # # Limpando o conjunto de teste
dados_ = estrato_treino_set.drop("median_house_value", axis=1)
dados_labels = estrato_treino_set["median_house_value"].copy()
# # print(dados)
# print(dados_labels)
# # # Lipando os dados
dados.dropna(subset=["total_bedrooms"])  # reitira os na
dados.drop("total_bedrooms", axis=1)  # Remove o atributo
mediana = dados["total_bedrooms"].median()
dados["total_bedrooms"] = dados["total_bedrooms"].fillna(mediana)
# # # Preenchendo valores ausentes pelo stic-klearn
imputa_mediana_na = SimpleImputer(strategy='median')
# # # retirando o atributo para calcuar a mediana
dados_num = dados.drop("ocean_proximity", axis=1)
imputa_mediana_na.fit(dados_num)
# print(imputa_mediana_na.statistics_)
# print(dados_num.median().values)
# # # Substituindo os valores ausentes pelas medianas treiandas
X = imputa_mediana_na.transform(dados_num)
# # # Colocar os dados no daframe
dados_treinado = pd.DataFrame(X, columns=dados_num.columns,
                              index=dados_num.index)
# print(dados_treinado)
# Trabalhando os Dados Categorizados
# Gerandoos dados categoricos
dados_categoricos = dados[["ocean_proximity"]]
print(dados_categoricos)
ordinal_encoder = OrdinalEncoder()
dados_categoricos_enconde = ordinal_encoder.fit_transform(dados_categoricos)
print(dados_categoricos_enconde[:17])
print(ordinal_encoder.categories_)
# Categorizar os dados pelo OneHotEnconder
cat_encoder = OneHotEncoder()
dados_categoricos_enconde_1hot = cat_encoder.fit_transform(dados_categoricos)
print(dados_categoricos_enconde_1hot)
# Modelo basico de transformadores para classes
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_househld = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_househld,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_househld]


att_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = att_adder.fit_transform(dados.values)
# Gerando o pipeline para transformação dos dados
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('sd_scaler', StandardScaler()),
])
# # Aplicando o pipeline de transformaçao
dados_num_tr = num_pipeline.fit_transform(dados_num)
print(dados_num, dados_num_tr[:2])
# Gerando as transformações no conjunto de dados completo
# Tanto nas variáveis numericas como nas categoricas
atributo_num = list(dados_num)
atributo_cat = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, atributo_num),
    ("cat", OneHotEncoder(), atributo_cat),
])
dados_prepara = full_pipeline.fit_transform(dados)
print(dados_prepara)
