# Gerando uma Função Para deixar os dados no Formato de Marchine Learning
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import kagglehub
from sklearn.linear_model import LinearRegression
def import_csv():
    data = pd.read_csv(
        'C:/Users/nonat/OneDrive/Desktop/Instituto Inteligência de Dados/Ciencia de Dados/DataScience/norway_new_car_sales_by_make.csv')
    data['Período'] = data['Year'].astype(
        'string') + '-' + data['Month'].astype(str).str.zfill(2)
    df = pd.pivot_table(data=data, values='Quantity',
                        index='Make', columns='Período', fill_value=0)
    return df

# Gerando o conjunto de treino e teste
def datasets(df, x_len=12, y_len=1, test_loops=12):
    D = df.values
    rows, Período = D.shape
    # Gerando o Conjunto de treinos
    loops = Período + 1 - x_len - y_len
    train = []
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)
    # Gerando o conjunto de Teste
    if test_loops > 0:
        # indexando o conjunto de treino e guardando uma parte para testas
        X_train, X_test = np.split(X_train, [-rows*test_loops], axis=0)
        Y_train, Y_test = np.split(Y_train, [-rows*test_loops], axis=0)
    else:  # No teste o conjunto: _teste é usado para gerar as prvisões futuras
        X_test = D[:, -x_len:]
        # Gerando o valor fake
        Y_test = np.full((X_test.shape[0], y_len), np.nan)
    # Preparando para usar a biblioteca scikit-learn
    if y_len == 1:
        # Reduz a dimensão para 1D pois a biblioteca scikitilear precisa desse formato
        Y_train = Y_train.ravel()
        # Reduz a dimensão para 1D pois a biblioteca scikitilear precisa desse formato
        Y_test = Y_test.ravel()
    return X_train, Y_train, X_test, Y_test
# Gerando os dados
df=import_csv()
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1,test_loops=12)
reg = LinearRegression() # Create a linear regression object
reg = reg.fit(X_train,Y_train) # Fit it to the training data
# Create two predictions for the training and test sets
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)
print(df.head(3))
#print(X_train)
#print(Y_train)
#print(X_test)
#print(Y_test)
print(Y_train_pred)
print(Y_test_pred)
# Estimando a eficiência do Modelo Através dos KPIs
def kpi_ML(Y_train, Y_train_pred,Y_test, Y_test_pred,name=''):
    df = pd.DataFrame(columns=['MAE','RMSE','Bias'],index=['Train','Test'])
    df.index.name=name
    df.loc['Train','MAE']=100*np.mean(abs(Y_train-Y_train_pred))/np.mean(Y_train)
    df.loc['Train','RMSE']=100*np.sqrt(np.mean((Y_train-Y_train_pred)**2))/np.mean(Y_train)
    df.loc['Train','Bias']=100*np.mean((Y_train-Y_train_pred))/np.mean(Y_train)
    df.loc['Test','MAE']=100*np.mean(abs(Y_test-Y_test_pred))/np.mean(Y_test)
    df.loc['Test','RMSE']=100*np.sqrt(np.mean((Y_test-Y_test_pred)**2))/np.mean(Y_test)
    df.loc['Test','Bias']=100*np.mean(np.mean(Y_test-Y_test_pred))/np.mean(Y_test)
    df = df.astype(float).round(1) # Ajustando em 1 casas decimais
    print(df)
kpi_ML(Y_train, Y_train_pred,Y_test, Y_test_pred,name='Regression')
# Gerando as previsões do Modelo
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1,test_loops=0)
reg = LinearRegression()
reg = reg.fit(X_train,Y_train)
forecast = pd.DataFrame(data=reg.predict(X_test).round(0), index=df.index)
forecast.to_csv('C:/Users/nonat/OneDrive/Desktop/Instituto Inteligência de Dados/Ciencia de Dados/DataScience/Modelo_Regressão_Python_ML/dados.csv',decimal='.')
print(forecast)