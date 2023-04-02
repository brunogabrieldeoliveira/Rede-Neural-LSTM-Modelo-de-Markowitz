# pip install yfinance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# carregando a base de dados

# TCC TESTES -----------
'''
tickers= ['SANB4.SA', 'TAEE11.SA', 'SAPR4.SA', 'FLRY3.SA', 'VALE3.SA',\
          'LREN3.SA', 'ROMI3.SA', 'ABEV3.SA', 'SULA11.SA', 'WEGE3.SA']
    
dt_ini, dt_fin= '2021-04-01', '2022-03-31' #1ANO 
ajusta_peso = True
peso_min, peso_max= 0.035, 0.15
qt_carteiras= 10000
'''

'''
# CARTEIRA TESTE INTERNACIONAL ----
tickers=['KO', 'PG']  

dt_ini, dt_fin= '2017-01-01', '2021-12-31'
ajusta_peso = True
peso_min, peso_max= 0.05, 0.15
qt_carteiras= 100000
'''


# CARTEIRA PESSOAL TESTES ------
tickers= ['ITSA4.SA', 'BBAS3.SA', 'SANB4.SA', 'EGIE3.SA', 'ENBR3.SA',\
          'ALUP11.SA', 'SAPR11.SA', 'CSMG3.SA', 'BBSE3.SA', 'WIZS3.SA']


dt_ini, dt_fin= '2017-06-01', '2022-05-31'
ajusta_peso = True
peso_min, peso_max= 0.03, 1
qt_carteiras= 100000


carteira= pd.DataFrame()
for tkr in tickers:
    carteira[tkr]= yf.download(tkr, start= dt_ini, end= dt_fin)['Close']
  
# plotando o grafico de variação diaria (normalizado)
sns.set()
(carteira / carteira.iloc[0] * 100).plot(figsize=(21,9))
#plt.show()


# SIMULAÇÃO ENTRADA EM SEMANAS ---

# # retorno semanal
# retorno_s = carteira.pct_change()
# retorno_s = retorno_s[-3:]

# # retorno anual
# retorno_a = retorno_s.mean() * 50

# # calculando a covariancia 
# cov= retorno_s.cov() * 50

# # calculando a correlação 
# corr = retorno_s.corr()



# SIMULAÇÃO ENTRADA EM DIAS ------

# retorno diario
retorno_d = carteira.pct_change()

# retorno anual
retorno_a = retorno_d.mean() * 250

# calculando a covariancia 
cov= retorno_d.cov() * 250

# calculando a correlação 
corr = retorno_d.corr()


# Simulando varias carteiras
lista_retorno, lista_volatilidade, lista_pesos, lista_sharpe_ratio= [], [], [], []

for carteiras in range(qt_carteiras):
    
    # calcula o peso
    peso= np.random.random(len(tickers))
    peso /= np.sum(peso)
    
    # calcula retorno esperado
    retorno_esperado= np.dot(peso, retorno_a)    

    # calcula volatilidade
    volatilidade= np.sqrt(np.dot(peso.T, np.dot(cov, peso)))

    # calcula sharpe_ration
    sharpe_ratio= retorno_esperado/volatilidade
    
    # salva cada calculo em sua respectiva lista
    lista_pesos.append(peso)
    lista_retorno.append(retorno_esperado)
    lista_volatilidade.append(volatilidade)
    lista_sharpe_ratio.append(sharpe_ratio)
    
dic_carteiras= {'Retorno': lista_retorno,
               'Volatilidade': lista_volatilidade,
               'Sharpe Ratio': lista_sharpe_ratio,
              }

for contar, acao in enumerate (tickers):
    dic_carteiras[acao]= [Peso[contar] for Peso in lista_pesos]
 
portfolios= pd.DataFrame(dic_carteiras)   
colunas= ['Retorno', 'Volatilidade', 'Sharpe Ratio'] + [acao for acao in tickers] 
portfolios= portfolios[colunas]

# AJUSTA PORTFÓLIOS CONFORME REGRAS ------

if ajusta_peso == True:
    
    for tk in tickers:
    
        # peso minimo
        portfolios= portfolios.loc[portfolios[tk] >= peso_min]
    
        # peso maximo
        portfolios= portfolios.loc[portfolios[tk] <= peso_max]

 
# PLOTA GRAFICOS -------


plt.style.use('seaborn-dark')
portfolios.plot.scatter(x='Volatilidade', 
                       y='Retorno', 
                       cmap='RdYlGn', 
                       edgecolors='black',
                       figsize=(21,9),
                       grid=True
                       )    
    
# menor volatilidade
carteira_volatilidade_menor= portfolios.loc[portfolios['Volatilidade'] == portfolios['Volatilidade'].min()]    

# maior volatilidade
carteira_volatilidade_maior= portfolios.loc[portfolios['Volatilidade'] == portfolios['Volatilidade'].max()]

# menor retorno
carteira_menor_retorno= portfolios.loc[portfolios['Retorno'] == portfolios['Retorno'].min()]   
  
# maior retorno
carteira_maior_retorno= portfolios.loc[portfolios['Retorno'] == portfolios['Retorno'].max()]
    
# maior indice de sharpe
carteira_sharpe_ratio_maior= portfolios.loc[portfolios['Sharpe Ratio'] == portfolios['Sharpe Ratio'].max()]  
    
    
    