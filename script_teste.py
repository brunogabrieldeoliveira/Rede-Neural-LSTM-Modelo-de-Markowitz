# IMPORTANDO BIBLIOTECAS --------------

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


# DEFININDO FUNÇÕES -------------------


# função coleta dados brutos 
def coleta_carteira(tickers, dt_ini, dt_fin):
       
    # dicionário armazenará os ativos da carteira
    carteira_coletada = {} 
    
    for tk in tickers:
        carteira_coletada[tk] = yf.download(tk, start=dt_ini, end=dt_fin) # download da série histórica de cada ativo
        carteira_coletada[tk] = carteira_coletada[tk].drop(columns=['Adj Close']) # remove coluna Adj Close da série histórica
    
    # retorna dicionário contendo todos os ativos da carteira
    return carteira_coletada 

# função trata dados brutos
def trata_carteira(tickers, carteira_coletada):
    
    carteira_tratada = {}
    
    for tk in tickers:
        carteira_tratada[tk] = carteira_coletada[tk] # faz a cópia da carteira coletada para a tratada
        carteira_tratada[tk] = carteira_tratada[tk].reset_index() # reseta o indice da tabela
        carteira_tratada[tk] = carteira_tratada[tk].drop(columns=['Date']) # exclui coluna data       
    
    # retorna dicionário de ativos contendo todos os ativos tratados
    return ajusta_serie_historica(tickers, carteira_tratada)

# função ajusta erros na série histórica
def ajusta_serie_historica(tickers, dados_tratados):
    
    # armazena dados brutos da carteira de ativos
    temp_dados = {}
    
    # cria cópia da carteira de ativos bruta
    for tk in tickers:        
        temp_dados[tk] = dados_tratados[tk]
        
    # Função verifica se a série histórica possui erros/furos na série histórica
    error = busca_erros(tickers, temp_dados)
    
    # Caso erro na série histórica, executa ajuste
    if error[0] == True:        
        for i in error[1]:
            # Valor do indice = (soma do valor anterior ao corrente + posterior ao corrente)/2
            temp_dados[i[0]][i[1]][i[2]] = (temp_dados[i[0]][i[1]][i[2]-1] + temp_dados[i[0]][i[1]][i[2] + 1])/2
                
    return temp_dados # retorna serie histórica tratada ou não

# função busca erros na série histórica
def busca_erros(tickers, dados):
    
    lista = [] # armazena lista contendo ticker, coluna, indice e valor de cada indice por ativo
    valida = False # se encontrar erros na série histórica o valor = true
    
    for tk in tickers: # varre cada ticker        
        for cl in ['Open', 'High', 'Low', 'Close', 'Volume']: # varre cada coluna de cada ticker   
                      
            indice = 0 # variavel para contagem dos indices
            for vl in dados[tk][cl]: # varre valores em cada ticker/coluna                
                if (vl == 0) or not(vl > 0):  # verdadeiro caso valor 0 ou nan                      
                    lista.append([tk, cl, indice, vl]) # adiciona valores a lista                 
                    valida = True  # caso ache erros retorna true                        
                indice = indice + 1 # incrementa indices                    
            indice = 0 # zera indices após varrer cada série temporal e carteira como um todo
            
    # retorna tupla contendo booleano achou erro e lista de indices com problemas
    return valida, lista 

# função ajuste de portfólio de markovitz
def ajusta_portfolio(tickers, 
                     temp_retornos, 
                     qt_carteiras, 
                     plota_grafico_ajuste_portfolio,
                     portfolio_tipo_retorno):
    
    # VARIAVEIS DE ENTRADA

    # retorno semanal
    retorno_s= temp_retornos

    # retorno anual
    retorno_a = retorno_s.mean() * 50

    # calculando a covariancia 
    cov= retorno_s.cov() * 50

    # calculando a correlação 
    corr = retorno_s.corr()
        
    # Simulando varias carteiras

    #qt_carteiras= 1000
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
               

    if plota_grafico_ajuste_portfolio == True:
        
        plt.style.use('seaborn-dark')
        portfolios.plot.scatter(x='Volatilidade', 
                                y='Retorno', 
                                cmap='RdYlGn', 
                                edgecolors='black',
                                figsize=(21,9),
                                grid=True
                                )
        
    return portfolios

# PARAMETROS --------------------------

# CARTEIRA NACIONAL ---------
    
#'''
# carteira de ativos
tickers= ['SANB4.SA', 'TAEE11.SA', 'SAPR4.SA', 'FLRY3.SA', 'VALE3.SA',\
          'LREN3.SA', 'ROMI3.SA', 'ABEV3.SA', 'SULA11.SA', 'WEGE3.SA']

# período de predição
dt_ini, dt_fin= '2021-04-01', '2022-03-31' #1ANO 

# simula carteira todos pesos iguais
pesos_iguais= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# inicia pesos iguais para carteira predicao
pesos= [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]

# pesos otimizados conforme teoria do portfólio
pesos_otimizados= [0.143, 0.142, 0.13, 0.052, 0.141, 0.035, 0.047, 0.1, 0.103, 0.107]

# Ajusta pesos minimo e maximo
ajusta_peso= True
peso_min, peso_max= 0.035, 0.15

# quantidade de carteiras simuladas por semana
qt_carteiras = 10000

#'''

# CARTEIRA INTERNACIONAL ----

'''
tickers=['KO', 'PG']  
dt_ini, dt_fin= '2017-01-01', '2021-12-31'

# pesos iguais para toda a carteira de ativos
pesos_iguais=[0.5, 0.5]

# inicia pesos iguais para todos os ativos (carteira predição)
pesos=[[0.5, 0.5]]

# pesos otimizados conforme teoria do portfólio
pesos_otimizados=[0.05, 0.95]

# Ajusta pesos minimo e maximo
ajusta_peso= True
peso_min, peso_max= 0.05, 0.15

# parametros ajuste portfólio
qt_carteiras = 10000
'''

# PARAMETROS GERAIS ---------

# preço de para análise e predição
entrada= 'Close'

# periodo futuro predição (em dias)
janela= 5

plota_grafico_ajuste_portfolio= False

# Tipo de retorno - Teoria do Portfólio
# 1 - Volatilidade Minima
# 2 - Volatilidade Maxima
# 3 - Retorno Minimo
# 4 - Retorno Maxima
# 5 - Maior sharpe Ratio

portfolio_tipo_retorno= 5

# caminho modelos de machine learning treinados
caminho= 'C:/Users/bruno/Documents/Trabalho_de_Conclusao_II/modelos_treinados/'


# COLETA SÉRIE DE PREÇOS --------------

# coleta ativos da carteira
base= trata_carteira(tickers, 
                      coleta_carteira(tickers, dt_ini, dt_fin)
                      )

# coleta ibovespa
ibov= trata_carteira(['^BVSP'], 
                      coleta_carteira(['^BVSP'], dt_ini, dt_fin)
                      )

# TRATA DADOS DA SÉRIE HISTÓRICA ------

# todos os ticker com a mesma quantidade de dias
for tk in tickers:
    temp_base= np.array(base[tk][entrada])
    temp_base= temp_base.reshape(-1, 1)
    
    # ajusta tamanho da série
    if (len(temp_base) % janela) > 0:
        temp_base= temp_base[:-(len(temp_base) % janela)] 
        
    base[tk]= temp_base
    
# cria dataframe carteira de ativos
carteira= pd.DataFrame(columns=(tickers))
lista_ativos= []

for tk in tickers:
    lista_ativos.append(pd.DataFrame(base[tk]))     

carteira= pd.concat(lista_ativos, axis= 1)
carteira.columns= tickers  
#carteira.index += 1

    
# IMPORTA MODELOS TREINADOS -----------

from keras.models import load_model

# importa modelos treinados
modelos_treinados= {}
        
for tk in tickers:
    modelos_treinados[tk] = load_model(caminho + tk + '/' + tk + '_model.h5')
    
 
# REALIZANDO A PREDIÇÃO ---------------


# normalizando dados
from sklearn.preprocessing import MinMaxScaler

# instancia a normalização dos dados
normalize = MinMaxScaler(feature_range=(0,1))

# carteira de ativos normalizada
carteira_normalizada= pd.DataFrame(normalize.fit_transform(carteira))
carteira_normalizada.columns= tickers 

# prepara carteira normalizada para predição
dic_carteira_normalizada= {}

for tk in tickers:    
    dic_carteira_normalizada[tk]= carteira_normalizada[tk].values.reshape(-1, janela, 1)


# cria carteira
dic_carteira_predicao= pd.DataFrame()

for tk in tickers:
    
    # lista predição semanal futura
    lista_predicao= []

    # roda todas as semanas para cada ticker
    for semana in dic_carteira_normalizada[tk]:        
        
        # pega a semana e ajusta shape dias 
        temp_semana= semana.reshape(1, janela, 1) 
        
        # realiza predicao da proxima sexta-feira (futuro)
        temp_predicao= modelos_treinados[tk].predict(temp_semana)
        temp_predicao= temp_predicao.reshape(1,1)
                
        # salva predicao semanal do ticker em uma lista
        lista_predicao.append(temp_predicao[0][0])
        
    # salva predição futura para cada ticker (monta no dataframe)
    dic_carteira_predicao[tk]= lista_predicao

# reverte normalização predição futura
dic_carteira_predicao= normalize.inverse_transform(dic_carteira_predicao)
carteira_predicao= pd.DataFrame(dic_carteira_predicao)
carteira_predicao.columns= tickers 
#carteira_predicao.index += 1


# LÓGICA DE PREDIÇÃO FUTURA -----------

carteira.index += 1
carteira_predicao.index += 1

cont= 0

conta_semana= 0

for indice in range(1, len(carteira)):
    
    # executa toda semana
    #if (indice % 5) == 0 and indice >= 5 or indice == (len(carteira)):
    if (indice % 5) == 0 and indice >= 5:
        
        # conta semanas         
        conta_semana = conta_semana + 1
        
        if indice >= 10:        
        
            # COLETANDO RETORNOS ------
        
            # retorno semanal passado
            temp_semana_passada= pd.DataFrame((carteira.iloc[indice -6]/carteira.iloc[indice - 10])-1)
            temp_semana_passada= temp_semana_passada.T
            
            # retorno semanal corrente
            temp_semana_corrente= pd.DataFrame((carteira.iloc[indice -1]/carteira.iloc[indice - 5])-1)
            temp_semana_corrente= temp_semana_corrente.T
         
            # retorno semanal futuro         
            temp_semana_futura= pd.DataFrame((carteira_predicao.iloc[conta_semana-1]/carteira.iloc[indice -1])-1)
            temp_semana_futura= temp_semana_futura.T          
                
         
            # DATAFRAME RETORNOS ------
        
            temp_retornos = pd.DataFrame(columns= tickers)         
            temp_retornos.loc[0]= temp_semana_passada.loc[0]
            temp_retornos.loc[1]= temp_semana_corrente.loc[0]
            temp_retornos.loc[2]= temp_semana_futura.loc[0]
       
            
            # AJUSTE DE PORTFOLIO -----
            
            temp_pesos= ajusta_portfolio(tickers, 
                                 temp_retornos,
                                 qt_carteiras,
                                 plota_grafico_ajuste_portfolio,
                                 portfolio_tipo_retorno
                                 )
            
            
            # AJUSTA PORTFÓLIOS CONFORME REGRAS ------

            if ajusta_peso == True:
                
                for tk in tickers:                    
                    temp_pesos = temp_pesos.loc[temp_pesos[tk] >= peso_min]
                    temp_pesos = temp_pesos.loc[temp_pesos[tk] <= peso_max]
                    
                    
            # RETORNOS POSSÍVEIS ---------------------         
        
         
            # menor volatilidade
            if portfolio_tipo_retorno == 1:
                temp_pesos= temp_pesos.loc[temp_pesos['Volatilidade'] == temp_pesos['Volatilidade'].min()]    

            # maior volatilidade
            if portfolio_tipo_retorno == 2:
                temp_pesos= temp_pesos.loc[temp_pesos['Volatilidade'] == temp_pesos['Volatilidade'].max()]

            # menor retorno
            if portfolio_tipo_retorno == 3:
                temp_pesos= temp_pesos.loc[temp_pesos['Retorno'] == temp_pesos['Retorno'].min()]   
  
            # maior retorno
            if portfolio_tipo_retorno == 4:
                temp_pesos= temp_pesos.loc[temp_pesos['Retorno'] == temp_pesos['Retorno'].max()]
    
            # maior indice de sharpe
            if portfolio_tipo_retorno == 5:
                temp_pesos= temp_pesos.loc[temp_pesos['Sharpe Ratio'] == temp_pesos['Sharpe Ratio'].max()]
            
                   
            pesos.append(temp_pesos[tickers].values[0])
        
            # print(temp_pesos)
            # print(pesos)
            # print(cont)
            # cont= cont + 1
            
    # ultima semana
    if indice == (len(carteira) -1):
        
        # conta semanas         
        conta_semana = conta_semana + 1
        
        # COLETANDO RETORNOS -----
        
        # retorno semanal passado
        temp_semana_passada= pd.DataFrame((carteira.iloc[indice -5]/carteira.iloc[indice - 9])-1)
        temp_semana_passada= temp_semana_passada.T
            
        # retorno semanal corrente
        temp_semana_corrente= pd.DataFrame((carteira.iloc[indice]/carteira.iloc[indice - 4])-1)
        temp_semana_corrente= temp_semana_corrente.T
        
        # retorno semanal futuro         
        temp_semana_futura= pd.DataFrame((carteira_predicao.iloc[conta_semana -1]/carteira.iloc[indice])-1)
        temp_semana_futura= temp_semana_futura.T
                
        # DATAFRAME RETORNOS -----
        
        temp_retornos = pd.DataFrame(columns= tickers)         
        temp_retornos.loc[0]= temp_semana_passada.loc[0]
        temp_retornos.loc[1]= temp_semana_corrente.loc[0]
        temp_retornos.loc[2]= temp_semana_futura.loc[0]
         
        
        # AJUSTE DE PORTFOLIO ---------
        
        
        temp_pesos= ajusta_portfolio(tickers, 
                                 temp_retornos,
                                 qt_carteiras,
                                 plota_grafico_ajuste_portfolio,
                                 portfolio_tipo_retorno
                                 )
        
        # AJUSTA PORTFÓLIOS CONFORME REGRAS ------

        if ajusta_peso == True:
            
            for tk in tickers:                    
                temp_pesos = temp_pesos.loc[temp_pesos[tk] >= peso_min]
                temp_pesos = temp_pesos.loc[temp_pesos[tk] <= peso_max]
        
        
        # RETORNOS POSSÍVEIS ---------------------         
        
         
        # menor volatilidade
        if portfolio_tipo_retorno == 1:
            temp_pesos= temp_pesos.loc[temp_pesos['Volatilidade'] == temp_pesos['Volatilidade'].min()]    

        # maior volatilidade
        if portfolio_tipo_retorno == 2:
            temp_pesos= temp_pesos.loc[temp_pesos['Volatilidade'] == temp_pesos['Volatilidade'].max()]

        # menor retorno
        if portfolio_tipo_retorno == 3:
            temp_pesos= temp_pesos.loc[temp_pesos['Retorno'] == temp_pesos['Retorno'].min()]   

        # maior retorno
        if portfolio_tipo_retorno == 4:
            temp_pesos= temp_pesos.loc[temp_pesos['Retorno'] == temp_pesos['Retorno'].max()]

        # maior indice de sharpe
        if portfolio_tipo_retorno == 5:
            temp_pesos= temp_pesos.loc[temp_pesos['Sharpe Ratio'] == temp_pesos['Sharpe Ratio'].max()]
        
        
        pesos.append(temp_pesos[tickers].values[0])
        
        # print(temp_pesos)
        # print(pesos)
        # print(cont)
        # cont= cont + 1
        
        
        
#transforma primeira lista de pesos em um array        
pesos[0] = np.array(pesos[0])


# ANALISE DE DADOS ------------


# normalizando a carteira predição
for tk in tickers:
    carteira_predicao[tk] = (carteira_predicao[tk] / carteira_predicao[tk].iloc[0] * 100) 


# criando carteira para análise (normalizada)
carteira_analise= pd.DataFrame()
carteira_analise.index += 1

for tk in tickers:
    carteira_analise[tk] = (carteira[tk] / carteira[tk].iloc[0] * 100)   

# sns.set()
# carteira_analise.plot(figsize=(21,9))  

# sns.set()   
# carteira_predicao.plot(figsize=(21,9))  


# cria carteira consolidada
lista_pesos_predicao= []
lista_pesos_otimizados= []
lista_pesos_iguais= []
conta_semana = 0
indice = 0


for indice in range(0, len(carteira_analise)):     
    
    # cria carteira pesos iguais
    lista_pesos_iguais.append(np.dot(carteira_analise.iloc[indice], pesos_iguais))
    
    # cria carteira pesos melhor otimização
    lista_pesos_otimizados.append(np.dot(carteira_analise.iloc[indice], pesos_otimizados))

    # cria carteira pesos predição
    if conta_semana <= (len(carteira_predicao)-1):        
        lista_pesos_predicao.append(np.dot(carteira_analise.iloc[indice], pesos[conta_semana]))
    
    # conta cada semana que passou
    if (indice % 5) == 0 and indice >= 5:        
        conta_semana = conta_semana + 1

'''
# PLOTA DADOS NACIONAIS ---------------

ibov= ibov['^BVSP']['Close']
ibov= (ibov/ibov.iloc[0] * 100)


# CARTEIRA PREDIÇÃO X IBOV ------------

carteira_predicao_ibov= pd.DataFrame(columns= ['IBOVESPA', 'PESOS_PREDICAO'])
carteira_predicao_ibov['IBOVESPA'] = ibov.iloc[:len(lista_pesos_predicao)] 
carteira_predicao_ibov['PESOS_PREDICAO']= lista_pesos_predicao

sns.set()
carteira_predicao_ibov.plot(figsize= (21,12))

# CARTEIRA PESOS IGUAIS X IBOV ------------

carteira_pesos_iguais_ibov= pd.DataFrame(columns= ['IBOVESPA', 'PESOS_IGUAIS'])
carteira_pesos_iguais_ibov['IBOVESPA'] = ibov.iloc[:len(lista_pesos_predicao)] 
carteira_pesos_iguais_ibov['PESOS_IGUAIS']= lista_pesos_iguais

sns.set()
carteira_pesos_iguais_ibov.plot(figsize= (21,12))

# CARTEIRA PESOS IGUAIS X PREDICAO X IBOV ------------

carteira_pesos_iguais_predicao_ibov= pd.DataFrame(columns= ['IBOVESPA', 'PESOS_PREDICAO', 'PESOS_IGUAIS'])
carteira_pesos_iguais_predicao_ibov['IBOVESPA'] = ibov.iloc[:len(lista_pesos_predicao)] 
carteira_pesos_iguais_predicao_ibov['PESOS_PREDICAO']= lista_pesos_predicao
carteira_pesos_iguais_predicao_ibov['PESOS_IGUAIS']= lista_pesos_iguais

sns.set()
carteira_pesos_iguais_predicao_ibov.plot(figsize= (21,12))
'''

# CARTEIRA PREDIÇÃO X CARTEIRA PESOS IGUAIS

carteira_predicao_pesos_iguais= pd.DataFrame(columns= ['PESOS_PREDICAO', 'PESOS_IGUAIS'])
carteira_predicao_pesos_iguais['PESOS_PREDICAO']= lista_pesos_predicao
carteira_predicao_pesos_iguais['PESOS_IGUAIS']= lista_pesos_iguais

sns.set()
carteira_predicao_pesos_iguais.plot(figsize= (21,12))

# CARTEIRA PREDIÇÃO X CARTEIRA PESOS OTIMIZADOS MARKOVITZ

carteira_predicao_pesos_otimizados = pd.DataFrame(columns= ['PESOS_PREDICAO', 'PESOS_OTIMIZADOS'])
carteira_predicao_pesos_otimizados['PESOS_PREDICAO']= lista_pesos_predicao
carteira_predicao_pesos_otimizados['PESOS_OTIMIZADOS']= lista_pesos_otimizados

sns.set()
carteira_predicao_pesos_otimizados.plot(figsize= (21,12))

# CARTEIRA PREDIÇÃO X CARTEIRA PESOS IGUAIS X CARTEIRA PESOS OTIMIZADOS MARKOVITZ

carteira_predicao_pesos_iguais_pesos_otimizados= pd.DataFrame(columns= ['PESOS_PREDICAO', 'PESOS_IGUAIS', 'PESOS_OTIMIZADOS'])
carteira_predicao_pesos_iguais_pesos_otimizados['PESOS_PREDICAO']= lista_pesos_predicao
carteira_predicao_pesos_iguais_pesos_otimizados['PESOS_IGUAIS']= lista_pesos_iguais
carteira_predicao_pesos_iguais_pesos_otimizados['PESOS_OTIMIZADOS']= lista_pesos_otimizados

sns.set()
carteira_predicao_pesos_iguais_pesos_otimizados.plot(figsize= (21,12))

'''

# CARTEIRA PREDIÇÃO X CARTEIRA PESOS IGUAIS X CARTEIRA PESOS OTIMIZADOS MARKOVITZ X IBOV

carteira_predicao_pesos_iguais_pesos_otimizados_ibov= pd.DataFrame(columns= ['IBOVESPA', 'PESOS_PREDICAO', 'PESOS_IGUAIS', 'PESOS_OTIMIZADOS'])
carteira_predicao_pesos_iguais_pesos_otimizados_ibov['IBOVESPA']= ibov.iloc[:len(lista_pesos_predicao)] 
carteira_predicao_pesos_iguais_pesos_otimizados_ibov['PESOS_PREDICAO']= lista_pesos_predicao
carteira_predicao_pesos_iguais_pesos_otimizados_ibov['PESOS_IGUAIS']= lista_pesos_iguais
carteira_predicao_pesos_iguais_pesos_otimizados_ibov['PESOS_OTIMIZADOS']= lista_pesos_otimizados

sns.set()
carteira_predicao_pesos_iguais_pesos_otimizados_ibov.plot(figsize= (21,12))

'''

# PLOTA DADOS INTERNACIONAIS ----------

'''
# coleta SP500
ivv= trata_carteira(['IVV'], 
                      coleta_carteira(['IVV'], dt_ini, dt_fin)
                      )
ivv= ivv['IVV']['Close']
ivv= (ivv/ivv.iloc[0] * 100)

# CARTEIRA PREDIÇÃO X IVV -------------
carteira_predicao_ivv= pd.DataFrame(columns= ['S&P500', 'PESOS_PREDICAO'])
carteira_predicao_ivv['S&P500'] = ivv.iloc[:len(lista_pesos_predicao)] 
carteira_predicao_ivv['PESOS_PREDICAO']= lista_pesos_predicao

sns.set()
carteira_predicao_ivv.plot(figsize= (21,12))


# CARTEIRA PREDIÇÃO X CARTEIRA PESOS OTIMIZADOS MARKOVITZ

carteira_predicao_pesos_otimizados = pd.DataFrame(columns= ['PESOS_PREDICAO', 'PESOS_OTIMIZADOS'])
carteira_predicao_pesos_otimizados['PESOS_PREDICAO']= lista_pesos_predicao
carteira_predicao_pesos_otimizados['PESOS_OTIMIZADOS']= lista_pesos_otimizados

sns.set()
carteira_predicao_pesos_otimizados.plot(figsize= (21,12))
'''

#ANALISE DE RESULTADOS ------

#TAXA DE RETORNO ANUAL DAS CARTEIRAS 

#CRIA DATAFRAMES CARTEIRAS
retorno_carteiras = pd.DataFrame()
retorno_carteiras['pesos_iguais'] = lista_pesos_iguais
retorno_carteiras['pesos_otimizados'] = lista_pesos_otimizados
retorno_carteiras['pesos_predicao'] = lista_pesos_predicao

#RETORNO DIÁRIO
retorno_carteiras['return_d_pesos_iguais'] = (retorno_carteiras['pesos_iguais'] / retorno_carteiras['pesos_iguais'].shift(1)) -1
retorno_carteiras['return_d_pesos_otimizados'] = (retorno_carteiras['pesos_otimizados'] / retorno_carteiras['pesos_otimizados'].shift(1)) -1
retorno_carteiras['return_d_pesos_predicao'] = (retorno_carteiras['pesos_predicao'] / retorno_carteiras['pesos_predicao'].shift(1)) -1

retorno_anual_pesos_iguais = retorno_carteiras['return_d_pesos_iguais'].mean() * 248
retorno_anual_pesos_otimizados = retorno_carteiras['return_d_pesos_otimizados'].mean() * 248
retorno_anual_pesos_predicao = retorno_carteiras['return_d_pesos_predicao'].mean() * 248

print('\n')
print('ANÁLISE DO RETORNO ANUAL DAS CARTEIRAS DE INVESTIMENTOS!')
print('\n')

print ('CARTEIRA PESOS IGUAIS= ' + str(round(retorno_anual_pesos_iguais, 3) * 100) + ' %')
print ('CARTEIRA PESOS OTIMIZADOS= ' + str(round(retorno_anual_pesos_otimizados, 3) * 100) + ' %')
print ('CARTEIRA PESOS PREDIÇÃO= ' + str(round(retorno_anual_pesos_predicao, 3) * 100) + ' %')


#TAXA DE RISCO ANUAL DAS CARTEIRAS 

risco_pesos_iguais= [retorno_carteiras[['return_d_pesos_iguais']].std() * 248 ** 0.5][0][0]
risco_pesos_otimizados= [retorno_carteiras[['return_d_pesos_otimizados']].std() * 248 ** 0.5][0][0]
risco_pesos_predicao= [retorno_carteiras[['return_d_pesos_predicao']].std() * 248 ** 0.5][0][0]

print('\n')
print('ANÁLISE DO RISCO ANUAL DAS CARTEIRAS DE INVESTIMENTOS!')
print('\n')


print ('CARTEIRA PESOS IGUAIS= ' + str(round(risco_pesos_iguais, 3) * 100) + ' %')
print ('CARTEIRA PESOS OTIMIZADOS= ' +str(round(risco_pesos_otimizados, 3) * 100) + ' %')
print ('CARTEIRA PESOS PREDIÇÃO= ' + str(round(risco_pesos_predicao, 3) * 100) + ' %')


print('\n')
print('COVARIANCIA E CORRELAÇÃO DAS CARTEIRAS DE INVESTIMENTOS!')
print('\n')

#COVARIANCIA E CORRELAÇÃO

retorno_d = pd.DataFrame()
retorno_d['pesos_iguais']= retorno_carteiras['return_d_pesos_iguais']
retorno_d['pesos_otimizados']= retorno_carteiras['return_d_pesos_otimizados']
retorno_d['pesos_predicao']= retorno_carteiras['return_d_pesos_predicao']


# covariancia
cov= retorno_d.cov() * 250

# correlação
corr = retorno_d.corr()


print('COVARIANCIA!')
print('\n')

#PESOS IGUAIS X PESOS OTIMIZADOS
print ('PESOS IGUAIS X PESOS OTIMIZADOS= ' + str(round(cov['pesos_iguais'][1], 4)))

#PESOS OTIMIZADOS X PESOS PREDICAO
print ('PESOS OTIMIZADOS X PESOS PREDICAO= ' + str(round(cov['pesos_otimizados'][2], 4)))

#PESOS PREDICAO X PESOS IGUAIS
print ('PESOS PREDICAO X PESOS IGUAIS= ' + str(round(cov['pesos_predicao'][0], 4)))



print('CORRELAÇÃO!')
print('\n')

#PESOS IGUAIS X PESOS OTIMIZADOS
print ('PESOS IGUAIS X PESOS OTIMIZADOS= ' + str(round(corr['pesos_iguais'][1], 4)))

#PESOS OTIMIZADOS X PESOS PREDICAO
print ('PESOS OTIMIZADOS X PESOS PREDICAO= ' + str(round(corr['pesos_otimizados'][2], 4)))

#PESOS PREDICAO X PESOS IGUAIS
print ('PESOS PREDICAO X PESOS IGUAIS= ' + str(round(corr['pesos_predicao'][0], 4)))


#MAIS TESTES

#CRIA DATAFRAMES CARTEIRAS
retorno_carteiras_2 = pd.DataFrame()
retorno_carteiras_2['pesos_iguais'] = lista_pesos_iguais[180:]
retorno_carteiras_2['pesos_otimizados'] = lista_pesos_otimizados[180:]
retorno_carteiras_2['pesos_predicao'] = lista_pesos_predicao[180:]


#RETORNO DIÁRIO
retorno_carteiras_2['return_d_pesos_iguais'] = (retorno_carteiras_2['pesos_iguais'] / retorno_carteiras_2['pesos_iguais'].shift(1)) -1
retorno_carteiras_2['return_d_pesos_otimizados'] = (retorno_carteiras_2['pesos_otimizados'] / retorno_carteiras_2['pesos_otimizados'].shift(1)) -1
retorno_carteiras_2['return_d_pesos_predicao'] = (retorno_carteiras_2['pesos_predicao'] / retorno_carteiras_2['pesos_predicao'].shift(1)) -1


risco_pesos_iguais_2= [retorno_carteiras_2[['return_d_pesos_iguais']].std() * 65 ** 0.5][0][0]
risco_pesos_otimizados_2= [retorno_carteiras_2[['return_d_pesos_otimizados']].std() * 65 ** 0.5][0][0]
risco_pesos_predicao_2= [retorno_carteiras_2[['return_d_pesos_predicao']].std() * 65 ** 0.5][0][0]


print ('CARTEIRA PESOS IGUAIS= ' + str(round(risco_pesos_iguais, 3) * 100) + ' %')
print ('CARTEIRA PESOS OTIMIZADOS= ' +str(round(risco_pesos_otimizados, 3) * 100) + ' %')
print ('CARTEIRA PESOS PREDIÇÃO= ' + str(round(risco_pesos_predicao, 3) * 100) + ' %')

retorno_carteiras_3= pd.DataFrame()
retorno_carteiras_3['pesos_iguais']=retorno_carteiras_2['pesos_iguais']
retorno_carteiras_3['pesos_otimizados']=retorno_carteiras_2['pesos_otimizados']
retorno_carteiras_3['pesos_predicao']=retorno_carteiras_2['pesos_predicao']

i= ['pesos_iguais', 'pesos_otimizados','pesos_predicao']
for t in i:
    retorno_carteiras_3[t] = (retorno_carteiras_3[t] / retorno_carteiras_3[t].iloc[0] * 100) 

sns.set()
retorno_carteiras_3.plot(figsize= (12,8))
