# Iniciando os modelos de Marchine Learning
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import kagglehub
# Puxar os dados
df = pd.read_csv(
    'C:/Users/nonat/OneDrive/Desktop/Instituto Inteligência de Dados/Ciencia de Dados/DataScience/norway_new_car_sales_by_make.csv')
# Montando o gráfico de barras para poder veririficar o comportamento dos dados
plt.bar(df['Year'], df['Quantity'], color='skyblue', edgecolor='black')
plt.xticks(df['Year'])
# Colocando os rótulos
plt.title('Venda Anual de Véiculos')
plt.xlabel('Ano')
plt.ylabel('Veículos')
plt.show()