# importando bibliotecas
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import macd_diff, sma_indicator #histograma macd e média móvel simples
from ta.momentum import rsi                   #rsi
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# função coleta dados brutos 
def coleta_carteira(tickers, dt_ini, dt_fin):
    
    # parametros -----------------------------------------
    # tickers: lista de tickers (list)
    # dt_ini:  data inicial da série histótica (texto)
    # dt_fin: data final da série histórica (texto)
    #
    # return: carteira de ativos bruta (tipo dicionário)
    # ----------------------------------------------------
    
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
        #carteira_tratada[tk] = carteira_tratada[tk].drop(columns='index') # exclui coluna index    
    
    # retorna dicionário de ativos contendo todos os ativos tratados
    return ajusta_serie_historica(tickers, carteira_tratada)

# função ajusta erros na série histórica
def ajusta_serie_historica(tickers, dados_tratados):
    
    # parametros -----------------------------------------
    # tickers: lista de tickers (list)
    # dados_tratados: carteira de ativos contendo a série histórica tratada
    #
    # return: retorna dicionário contendo série histórica tratada ou não
    # ----------------------------------------------------
    
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
    
    # parametros -----------------------------------------
    # tickers: lista de tickers (list)
    # dados: carteira de ativos contendo a série histórica tratada
    #
    # return: retorna tupla contendo valor booleano e lista de ativos/indices que apresentam erros
    # ----------------------------------------------------
    
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

# função prepara a base de treino
def prepara_base_treino(base, janela):
    x_train, y_train = [], []

    for indice in range(janela, len(base)):
        x_train.append(base[indice - janela: indice, 0])
        y_train.append(base[indice, 0]) #*    
    
    x_train, y_train= np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
    
    return x_train, y_train

# função prepara a base de treino
def prepara_base_treino_2(base, janela, futuro):
    x_train, y_train = [], []

    for indice in range(janela, len(base)):
        x_train.append(base[indice - janela: indice, 0])    
        if indice < len(base) - futuro - 1:
            y_train.append(base[indice + futuro - 1, 0]) #*    
    
    x_train, y_train= np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
 
    x_train= x_train[:len(y_train)] 
    
    return x_train, y_train

# função prepara a base de teste
def prepara_base_teste(modelo_entrada, janela):
    x_teste = []
    for indice in range(janela, len(modelo_entrada)):
        x_teste.append(modelo_entrada[indice - janela: indice, 0])
    
    x_teste = np.array(x_teste)
    x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))
    
    return x_teste

# função cria a rede neural
def treina_rede_neural(x_train, y_train, neuronio, ativacao, dropout, optimizador, perda, epocas, lotes):
    
    # instancia o modelo
    model= Sequential()

    model.add(LSTM(units= neuronio, activation= ativacao, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units= neuronio, activation= ativacao, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units= neuronio, activation= ativacao, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units= neuronio, activation= ativacao))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1, activation= ativacao)) 
    
    # compila o modelo
    model.compile(
        optimizer= optimizador, 
        loss=      perda
        #metrics=   'accuracy'
        )
    # treina o modelo
    model.fit(
        x_train, 
        y_train, 
        epochs = epocas, 
        batch_size = lotes
        )    
        
    
    return model

# função retorno o modelo de entrada
def cria_modelo_entrada(base_total, dados_teste, janela, normalize):
    
    modelo_entrada = base_total[len(base_total) - len(dados_teste) - janela:].values
    modelo_entrada = modelo_entrada.reshape(-1, 1)

    # retorna modelo de entrada normalizado
    return normalize.fit_transform(modelo_entrada)

# função para exibir o grafico de preçoes reais e previstos
def exibe_grafico_previsao(ticker, preco_real, previsao):
    plt.plot(preco_real, color='red', label= f'Preço real {ticker}') 
    plt.plot(previsao, color='green', label= f'Previsao preços {ticker}')
    plt.title(f'{ticker}') 
    plt.xlabel('Tempo')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()
    
# função para prever o dia futuro
def previsao_futura(modelo, modelo_entrada, janela):
    #dados_reais= [modelo_entrada[len(modelo_entrada) + 1 - janela:len(modelo_entrada + 1), 0]]
    dados_reais= [modelo_entrada[len(modelo_entrada) - janela:len(modelo_entrada + 1), 0]]
    dados_reais= np.array(dados_reais)
    dados_reais= np.reshape(dados_reais, (dados_reais.shape[0], dados_reais.shape[1], 1))
    
    previsao= modelo.predict(dados_reais)
    previsao= normalize.inverse_transform(previsao)

    print('\n')
    print(f'Proxima previsão: {previsao}')        
    
# PARAMETRIZANDO A REDE NEURAL ---------------


# todas os tickers

tickers= ['SANB4.SA', 'TAEE11.SA', 'SAPR4.SA', 'FLRY3.SA', 'VALE3.SA',\
          'LREN3.SA', 'ROMI3.SA', 'ABEV3.SA', 'SULA11.SA', 'WEGE3.SA']


#tickers= ['SULA11.SA', 'WEGE3.SA']
#tickers= ['SANB4.SA', 'TAEE11.SA']
#tickers= ['SAPR4.SA', 'FLRY3.SA']
#tickers= ['VALE3.SA', 'LREN3.SA']
#tickers= ['ROMI3.SA', 'ABEV3.SA']

dt_ini, dt_fin= '2013-01-01', '2018-12-31'               #5 ANOS
dt_ini_teste, dt_fin_teste= '2019-01-01', '2020-12-31'   #2 ANOS


entrada= 'Close'

# dias futuros para prever
futuro= 5

# caminho exporta/importa modelos treinados
caminho= 'C:/Users/bruno/Documents/Trabalho_de_Conclusao_II/modelos_treinados/'

# caminho para importação dos parametros de treinamento da rede
caminho_parametros= 'C:/Users/bruno/Documents/Trabalho_de_Conclusao_II/'
arquivo= 'importa_parametros.csv'

# Importa ou nao parametros para a rede neral
importa_param= True

# exporta o treinamento de menor erro quadratico médio em .h5
exporta_treinamento= True

# importa o treinamento de menor erro quadratico médio em .h5
importa_treinamento= False

# Exibe graficos e previsão futura
exibe_grafico, exibe_previsao= False, False


# COLETANDO A BASE DE DADOS ------------------

# importa dados csv 
#dados = pd.read_csv('C:/Users/bruno/Documents/trabalho_conclusao_2/TAEE11.SA.csv', sep=";")

# coletando carteira de ativos tratada
dados= trata_carteira(tickers, 
                      coleta_carteira(tickers, dt_ini, dt_fin)
                      )

# coletando a base de teste tratada
dados_teste= trata_carteira(tickers,
                            coleta_carteira(tickers, dt_ini_teste, dt_fin_teste)
                            )

# Busca por erros na série histórica tratada
#error1 = busca_erros(tickers, dados)
#error2 = busca_erros(tickers, dados_teste)

# Instancia a normalização dos dados
normalize = MinMaxScaler(feature_range=(0,1))


# INICIANDO TRABALHOS DE PREDIÇÃO ------------

# importa parametros csv 
#param_rnn = pd.read_csv('C:/Users/bruno/Documents/trabalho_conclusao/param_rnn_teste.csv', sep=",")    
param_rnn = pd.read_csv(caminho_parametros + arquivo, sep=",")    
param_rnn = param_rnn.values

# dicionario melhores modelos treinados
dic_treino_menor_erro= {}      
    
for tk in tickers:
    
    # listas para salvar os dados de treinamento de cada ativo
    treinamento= []
    lista_erros= []
    lista_param= []
            
    for param in param_rnn:
        
        janela=      param[0]
        neuronio=    param[1]
        ativacao=    param[2]
        dropout=     param[3]
        optimizador= param[4]
        perda=       param[5]
        epocas=      param[6]
        lotes=       param[7]   
        
        # salva parametros        
        parametros= str(janela) + ', ' + str(neuronio) + ', ' + str(ativacao)\
           + ', ' + str(dropout) + ', ' + str(optimizador)\
           + ', ' + str(perda) + ', ' + str(epocas) + ', ' + str(lotes)       
    
        # PREPARANDO A BASE DE TREINO E TESTE --------
    
        ativo_treino= dados[tk]         # dados treino
        ativo_teste= dados_teste[tk]    # dados teste


        # BASE TREINO ---------
                
        # preparando os dados
        base = ativo_treino[entrada].values.reshape(-1, 1)

        # normalizando a base de dados de treino
        base = normalize.fit_transform(base)

        # # prepara a base de treino a base de treino
        # x_train, y_train= prepara_base_treino(base, janela)
            
        # prepara a base de treino a base de treino
        x_train, y_train= prepara_base_treino_2(base, janela, futuro)

        # BASE TESTE ----------

        # preparando os dados base teste
        preco_real = ativo_teste[entrada].values

        # dados totais
        base_total = pd.concat((ativo_treino[entrada], ativo_teste[entrada]), axis= 0)

        # criando o modelo de entrada
        modelo_entrada= cria_modelo_entrada(base_total, ativo_teste, janela, normalize)

        # prepara a base de teste
        x_teste= prepara_base_teste(modelo_entrada, janela)


        # TREINANDO A REDE NEURAL --------------------
    
        print('\n')
        print(f'Iniciando treinamento para {tk}')    
        print('\n')
            
        print('Parametros -----')
          
        print('janela, neuronios, ativacao, dropout, otimizador, perda, epocas, lotes')            
        print(f'{parametros}')
        print('\n')
          
        print('Treinando ...')
        print('\n')
    
        # cria e treina a rede neural
        modelo= treina_rede_neural(
            x_train, 
            y_train,
            neuronio,
            ativacao,
            dropout,
            optimizador,
            perda,
            epocas,
            lotes
            )            

        # FAZENDO AS PREVISOES -----------------------

        # prevendo os preços
        previsao = modelo.predict(x_teste)
          
        # normalização reversa para converter para os preços originais
        previsao = normalize.inverse_transform(previsao)
        
        # obtem o erro quadratico médio
        erro_quadratico_medio= mse(preco_real, previsao)            
            
            
        print('\n')            
        print('MSE: ', erro_quadratico_medio)

        # plota grafico
        if exibe_grafico == True:
            exibe_grafico_previsao(tk, preco_real, previsao)

        # previsao futura
        if exibe_previsao == True:
            previsao_futura(modelo, modelo_entrada, janela)
                
        # GRAVANDO OS MODELOS TREINADOS --------
        
        # salva lista de parametros
        lista_param.append(parametros)
                
        # salva a lista de erros
        lista_erros.append(erro_quadratico_medio)
            
        # salva a lista de modelos treinados                     
        treinamento.append(modelo)
                

    # SALVANDO OS MELHORES MODELOES ----------
        
    # converte lista erros para dataframe
    lista_erros= pd.DataFrame(lista_erros)
        
    # obtem o menor erro
    menor_erro= lista_erros.min().item()

    # obtem o indice que contém o menor erro
    indice_menor_erro=  lista_erros.idxmin().item()
    
    # obtem a lista de parametros referente ao menor erro
    parametros_menor_erro= lista_param[indice_menor_erro]

    # obtem o modelo de menor erro
    modelo_menor_erro= treinamento[indice_menor_erro]

    # salva somente o modelo treinado que obteve o menor erro
    dic_treino_menor_erro[tk]= [modelo_menor_erro, menor_erro, parametros_menor_erro]
        

# # testa se modelo foi salvo
# dic_treino_menor_erro['ROMI3.SA']
# dic_treino_menor_erro['ROMI3.SA'][0] # modelo
# dic_treino_menor_erro['ROMI3.SA'][1] # erro

# EXPORTANDO/IMPORTANDO MODELOS TREINADOS ------------

# exportando e importando modelos treinados
from keras.models import load_model

# Salvando o modelo treinado
if exporta_treinamento == True:  
    for tk in tickers:    
        dic_treino_menor_erro[tk][0].save(caminho + tk + '/' + tk + '_model.h5')
        
        with open(caminho + tk + '/info.txt', 'w') as arquivo:
            arquivo.write(f'Informações do modelo treinado {tk}')
            
            arquivo.write('Periodo de treino')
            arquivo.write('\n')
            arquivo.write('Incial: ' + str(dt_ini) + ';')
            arquivo.write('\n')
            arquivo.write('Final: ' + str(dt_fin) + ';')
            arquivo.write('\n')
            arquivo.write('\n')        
            arquivo.write('Menor erro: ' + str(dic_treino_menor_erro[tk][1]) + ' (em centavos)')
            arquivo.write('\n')
            arquivo.write('\n')
            arquivo.write('Parametros utilizados:')            
            arquivo.write('\n')            
            arquivo.write('janela, neuronio, ativacao, dropout, otimizador, perda, epocas, lotes')
            arquivo.write('\n')
            arquivo.write(str(dic_treino_menor_erro[tk][2]))
            arquivo.write('\n')
            arquivo.write('\n')
    
            arquivo.write('//--------------')  
                
# salva os modelos importados, caso importa_treinamento for verdadeiro
modelos_treinados= {}
        
# importa cada um dos modelos treinados
if importa_treinamento == True:    
    
    for tk in tickers:
        modelos_treinados[tk] = load_model(caminho + tk + '/' + tk + '_model.h5')

# # testa se os modelos foram importados com sucesso
# modelos_treinados['ROMI3.SA']
    



        


