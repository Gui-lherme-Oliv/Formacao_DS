<div align="justify">
  
# Resolução de exercícios práticos de Data Science utilizando Python
#### Autor: Guilherme Oliveira da Rocha Cunha

Estes exercícios foram apresentados no curso **Formação Cientista de Dados: O Curso Completo** oferecido pela plataforma de cursos online [Udemy](https://www.udemy.com/pt/). Este é um curso de Data Science em que se conhece e aprende a aplicar todos os principais conceitos e técnicas para se qualificar e atuar como um Cientista de Dados, com videos explicativos e detalhados, exemplos práticos de codificação em R e Python usando dados reais e explicações de resolução de fórmulas passo a passo. A resolução de todos os exercícios foi feita por mim, construída de acordo com as instruções dos professores Fernando Amaral e Jones Granatyr, instrutores responsáveis por este curso.

Os conjuntos de dados disponibilizados pela Udemy estão presentes neste repositório, no formato .csv dentro da pasta "dados". Já a resolução dos exercícios estão presentes aqui mesmo, para que tenha uma visualização mais rápida e fácil.

## Sumário
#### [1. Introdução ao Python](#1-introdução-ao-python-1)
#### [2. Limpeza e Tratamento de dados](#2-limpeza-e-tratamento-de-dados-1)
#### [3. Gráficos, Visualização e Dashboards](#3-gráficos-visualização-e-dashboards-1)
#### [4. Estatística I](#4-estatística-i-1)
#### [5. Estatística II](#5-estatística-ii-1)
#### [6. Regressão Linear](#6-regressão-linear-1)
#### [7. Regressão Logística](#7-regressão-logística-1)
#### [8. Séries Temporais](#8-séries-temporais-1)
#### [9. Machine Learning](#9-machine-learning-1)
#### [10. Neural Networks e Deep Learning](#10-neural-networks-e-deep-learning-1)
#### [11. Grafos](#11-grafos-1)
#### [12. Processamento de Linguagem Natural e Mineração de Texto](#12-processamento-de-linguagem-natural-e-mineração-de-texto-1)
#### [13. Bancos de Dados e Linguagem SQL](#13-bancos-de-dados-e-linguagem-sql-1)
#### [14. Bancos de Dados NoSQL e MongoDB](#14-bancos-de-dados-nosql-e-mongodb-1)
#### [15. Computação na Nuvem](#15-computação-na-nuvem-1)
#### [16. Spark com Databricks](#16-spark-com-databricks-1)  
---
## 1. Introdução ao Python
1. Faça um programa que tenha uma função chamada amplitude. A função deve receber uma lista e imprimir a amplitude.
Crie também um código para testar sua função
```
#RESOLUÇÃO

def amplitude(vet):
    print("Amplitude:", max(vet) - min(vet))

vetor = [12,23,45,2,100]    
amplitude(vetor)
```

2. Faça uma função que receba uma string e imprima esta string na forma vertical
Por exemplo, se receber python, deve imprimir  
p  
y  
t     
h  
o  
n  
Dica: uma string do python funciona como uma lista!
Crie também um código para testar sua função
```
#RESOLUÇÃO

def imprime(texto):
    for n in range(0, len(texto)):
        print(texto[n])

imprime("Guilherme")
```

3. Crie um programa que leia o peso de uma carga em números inteiros. Se o peso for até 10 kg, informe
que o valor será de R$ 50,00. Entre 11 e 20 kg, informe que o valor será de R$ 80. Se for maior que 20
informe que o transporte não é aceito. Teste vários pesos.
```
#RESOLUÇÃO

peso = 10
if peso <= 10:
	print("Valor da carga é de R$ 50,00")
elif peso >= 11 and peso <=20:
	print("Valor da carga é de R$ 80,00")
else:
	print("O transporte não é aceito")
```
#### [Voltar ao Sumário](#sumário)

## 2. Limpeza e Tratamento de Dados
Tratar dados "tempo.csv". Faça cada etapa de forma separada e  lembre-se de tratar os valores NaNs.

Domínio dos atributos:
- Aparência: sol, nublado, chuva
- Temperatura: -135 ~ 130 F
- Umidade: 0 ~ 100
- Jogar: sim/nao
```
#RESOLUÇÃO

import pandas as pd
import seaborn as srn
import statistics  as sts

# Importando dados
dataset = pd.read_csv("tempo.csv", sep=";")
#visualizar
dataset.head()

# Explorando dados categóricos
#Aparencia
agrupado_aparencia = dataset.groupby(['Aparencia']).size()
agrupado_aparencia.plot.bar(color = 'gray')

# Vento
agrupado_vento = dataset.groupby(['Vento']).size()
agrupado_vento.plot.bar(color = 'gray')

# Jogar
agrupado_jogar = dataset.groupby(['Jogar']).size()
agrupado_jogar.plot.bar(color = 'gray')

# Explorando dados numéricos
#Temperatura
dataset['Temperatura'].describe()

# Umidade
dataset['Umidade'].describe()

# Contando valores NAN
dataset.isnull().sum()

# Substituindo o valor inválido "menos" pelo valor "sol"
dataset.loc[dataset['Aparencia'] ==  'menos', 'Aparencia'] = "sol"
#visualiza o resultado
agrupado = dataset.groupby(['Aparencia']).size()
agrupado

# Visualizando os valores de "Temperatura" que estão fora do domínio
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130) ]

# Calculando a mediana e substituindo o valor fora do domínio por ela
mediana = sts.median(dataset['Temperatura'])
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130), 'Temperatura'] = mediana

# Umidade, dominio e NaNs
agrupado = dataset.groupby(['Umidade']).size()
agrupado

# Total de NaNs
dataset['Umidade'].isnull().sum()

# Calculando a mediana e preenchendo os NaNs
mediana = sts.median(dataset['Umidade'])
dataset['Umidade'].fillna(mediana, inplace=True)

# Total de NaNs para "Vento"
dataset['Vento'].isnull().sum()

# Preenchendo com "FALSO" (que é a maior ocorrência)
dataset['Vento'].fillna('FALSO', inplace=True)
```
#### [Voltar ao Sumário](#sumário)

## 3. Gráficos, Visualização e Dashboards
1. Arquivo "dados.csv":
- CODIGO
- MUNICIPIO
- PIB
- VALOREMPENHO
2. Mostre os municípios com maiores valores de PIB e de EMPENHO
```
#RESOLUÇÃO

# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as srn

# Carregando a base de dados
base = pd.read_csv('dados.csv', sep=';')
base.shape

base.head()

# Criando histograma da variável "PIB"
srn.histplot(base.PIB, kde=True, bins=10).set(title='PIB')

# Criando histograma da variável "VALOREMPENHO"
srn.histplot(base.VALOREMPENHO, kde=True, bins=10).set(title='Empenho')

# Gráfico dos 10 municípios com os maiores valores de PIB
agrupado = base.sort_values('PIB', ascending=False).head(10)
agrupado = agrupado.iloc[:, 1:3]
agrupado
agrupado.plot.bar(x='MUNICIPIO', y='PIB', color='gray')

# Gráfico dos 10 municípios com os maiores VALOREMPENHO
agrupado = base.sort_values('VALOREMPENHO').head(10)
agrupado = agrupado.iloc[:,[1,3]]
agrupado
agrupado.plot.bar(x='MUNICIPIO',y='VALOREMPENHO', color = 'gray')

```
#### [Voltar ao Sumário](#sumário)

## 4. Estatística I
Construa exemplos de:  

#### 4.1. Amostragem simples
Dados: "iris.csv"

```
# Importando as bibliotecas: pandas para carregar arquivos .csv e numpy para gerar números aleatórios
import pandas as pd
import numpy as np

# Carregando a base de dados e visualização dos dados
base = pd.read_csv('iris.csv')
base

# Verificando quantas linhas e colunas existem na base de dados, 150 linhas e 5 colunas
base.shape

# Mudando a semente aleatória randômica para manter os resultados em várias execuções
np.random.seed(2345)

# Amostra aleatória de 150 valores, valores de 0 a 1, com reposição, probabilidades equivalentes
# (p = [0.7, 0.3] define as probabilidades de ocorrência dos valores 0 e 1)
amostra = np.random.choice(a = [0, 1], size = 150, replace = True, p = [0.7, 0.3])
amostra

# Selecionando apenas as linhas onde o valor correspondente em amostra é igual a 0
base_final = base.loc[amostra == 0]
base_final.shape

# Selecionando apenas as linhas onde o valor correspondente em amostra é igual a 1
base_final2 = base.loc[amostra == 1]
base_final2.shape
```

#### 4.2. Medidas de Centralidade e Variabilidade
```
# Importando as bibliotecas: scipy para gerar estatísticas mais detalhadas
import numpy as np
from scipy import stats

# Criando a variável com os dados dos jogadores, visualização da média e mediana
jogadores = [40000, 18000, 12000, 250000, 30000, 140000, 300000, 40000, 800000]
media = np.mean(jogadores)
mediana = np.median(jogadores)

# Criando a variável para geração dos quartis (0%, 25%, 50%, 75% e 100%) 
quartis = np.quantile(jogadores, [0, 0.25, 0.5, 0.75, 1])
quartis

# Visualização do desvio padrão
# Quando ddof = 1 é especificado, o cálculo passa a ser o do desvio padrão amostral, usando o denominador N - 1.
# Isso ajusta o cálculo para amostras de uma população, corrigindo o viés da estimativa.
np.std(jogadores, ddof = 1)

# Visualização de estatísticas mais detalhadas usando a biblioteca scipy
stats.describe(jogadores)
```

#### 4.3. Distribuição Normal
```
# Importando da função norm
from scipy.stats import norm

# Conjunto de objetos em uma cesta, a média é 8 e o desvio padrão é 2
# Qual a probabilidade de tirar um objeto que peso é menor que 6 quilos?
norm.cdf(6, 8, 2)

# Qual a probabilidade de tirar um objeto que o peso á maior que 6 quilos?
1 - norm.cdf(6, 8, 2)

# Qual a probabilidade de tirar um objeto que o peso é menor que 6 ou maior que 10 quilos?
# A função sf (survival function) é equivalente a 1 - cdf
norm.cdf(6, 8, 2) + norm.sf(10, 8, 2)

# Qual a probabilidade de tirar um objeto que o peso é menor que 10 e maior que 8 quilos?
norm.cdf(10, 8, 2) - norm.cdf(8, 8, 2)
```
#### [Voltar ao Sumário](#sumário)

## 5. Estatística II
Construa exemplos de:  

#### 5.1. Distribuição T de Student
```
# Importando a função para fazer o teste
from scipy.stats import t

# Considerando que a média de salário dos cientistas de dados = R$ 75,00 por hora; Amostra com 9 funcionários e desvio padrão = 10
# O t-score encontrado na tabela para esse caso é de 1,5

# Qual a probabilidade de selecionar um cientista de dados e o salário ser menor que R$ 80 por hora
t.cdf(1.5, 8)

# Qual a probabilidade do salário ser maior do que 80?
t.sf(1.5, 8)

# Somatório da execução dos dois códigos acima (lado esquerdo + lado direito da distribuição)
t.cdf(1.5, 8) + t.sf(1.5, 8)
```

#### 5.2. Distribuição de Poisson
```
# Importando a função
from scipy.stats import poisson

# Considerando que a média de acidentes de carro é 2 por dia

# Qual a probabilidade de ocorrerem 3 acidentes no dia?
poisson.pmf(3, 2)

# Qual a probabilidade de ocorrerem 3 ou menos acidentes no dia?
poisson.cdf(3, 2)

# Qual a probabilidade de ocorrerem mais de 3 acidentes no dia?
poisson.sf(3, 2)

# Calculando todas as probabilidades (3 ou menos acidentes no dia + mais de 3 acidentes no dia)
poisson.cdf(3, 2) + poisson.sf(3, 2)
```

#### 5.3. Distribuição Binomial
```
# Importando a função binom
from scipy.stats import binom

# Jogar uma moeda 5 vezes, qual a probabilidade de dar cara 3 vezes?
# eventos , experimentos, probabilidades
prob = binom.pmf(3, 5, 0.5)
prob

# Passar por 4 sinais de 4 tempos, qual a probabilidade de pegar sinal verde
# nenhuma, 1, 2, 3 ou 4 vezes seguidas?
binom.pmf(0, 4, 0.25) + binom.pmf(1, 4, 0.25) + binom.pmf(2, 4, 0.25) + binom.pmf(3, 4, 0.25) + binom.pmf(4, 4, 0.25)

# E se forem sinais de dois tempos?
binom.pmf(4, 4, 0.5)

# Probabilidade acumulativa
binom.cdf(4, 4, 0.25)

# Concurso com 12 questões, qual a probabilidade de acertar 7 questões considerando
# que cada questão tem 4 alternativas?
binom.pmf(7, 12, 0.25)

# Probabilidade de acertar as 12 questões
binom.pmf(12, 12, 0.25)
```

#### 5.4. Qui-Quadrado
```
# Importação das funções, chi2_contingency porque são 2 categorias
import numpy as np
from scipy.stats import chi2_contingency

# Se existe diferença significativa entre homens e mulheres que assistem ou não novela

# Criação da matriz com os dados e execução do teste
# H: 19 sim e 6 não; M: 43 sim e 32 não
novela = np.array([[19, 6], [43, 32]])
novela

# Segundo valor no resultado é o p-value
# Se o p-value é maior que alfa, então não há evidências de diferença significativa (hipótese nula): não há diferença significativa
chi2_contingency(novela)

# com outros valores
novela2 = np.array([[22, 3], [43, 32]])
novela2

# Se o valor de p for menor, pode-se rejeitar a hipótese nula em favor da hipótese alternativa: há diferença significativa
chi2_contingency(novela2)
```
#### [Voltar ao Sumário](#sumário)

## 6. Regressão Linear
Dados: "slr12.csv"

1. Franquias
- FrqAnual: Taxa Anual
- CusInic: Investimento Inicial

2. Criar um modelo de regressão linear para prever qual será o Investimento inicial necessário de uma franquia a partir da Taxa Anual cobrado pelo franqueador.

```
#RESOLUÇÃO

# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando a base de dados
base = pd.read_csv('slr12.csv', sep=';')

# Definindo as variáveis X e y, X FrqAnual é a variável independente e y CusInic é a variável dependente
X = base['FrqAnual'].values.reshape(-1, 1)  # Reshape para formato de matriz
y = base['CusInic'].values

# Calculando a correlação entre X e y
correlacao = np.corrcoef(X.flatten(), y)
print(f"Correlação: {correlacao[0, 1]}")

# Criação do modelo e treinamento
modelo = LinearRegression()
modelo.fit(X, y)

# Fazendo previsões
y_pred = modelo.predict(X)

# Calculando R² e erro quadrático médio
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"R²: {r2}")
print(f"MSE: {mse}")

# Gerando o gráfico com os pontos reais e as previsões
plt.scatter(X, y, label='Dados Reais')
plt.plot(X, y_pred, color='red', label='Regressão Linear')
plt.title('Regressão Linear: Investimento Inicial vs Taxa Anual')
plt.xlabel('Taxa Anual (%)')
plt.ylabel('Investimento Inicial (R$)')
plt.legend()
plt.grid()
plt.show()
```
#### [Voltar ao Sumário](#sumário)

## 7. Regressão Logística
Dados: "Eleicao.csv"; "NovosCandidatos.csv"

1. Criar um modelo logístico para a relação entre o investimento de um candidato na campanha para um cargo legislativo e o fato dele ser eleito ou não.
2. Com o modelo criado, fazer previsões para novos candidatos.

```
#RESOLUÇÃO

# Importando as bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Carregando a base de dados, visualização de gráfico com os pontos e visualização de estatísticas
base = pd.read_csv('Eleicao.csv', sep = ';')
plt.scatter(base.DESPESAS, base.SITUACAO)
base.describe()

# Visualizando o coeficiente de correlação entre o atributo "despesas" e "situação"
np.corrcoef(base.DESPESAS, base.SITUACAO)

# Criando as variávies X e y (variável independente e variável dependente)
# Transformação de X para o formato de matriz adicionando um novo eixo (newaxis)
X = base.iloc[:, 2].values
X = X[:, np.newaxis]
y = base.iloc[:, 1].values

# Criação do modelo, treinamento e visualização dos coeficientes
modelo = LogisticRegression()
modelo.fit(X, y)
modelo.coef_
modelo.intercept_

plt.scatter(X, y)
# Gerando novos dados para gerar a função sigmoide
X_teste = np.linspace(10, 3000, 100)
# Implementando da função sigmoide
def model(x):
    return 1 / (1 + np.exp(-x))
# Gerando previsões (variável r) e visualização dos resultados
r = model(X_teste * modelo.coef_ + modelo.intercept_).ravel()
plt.plot(X_teste, r, color = 'red')

# Carregando a base de dados com os novos candidatos
base_previsoes = pd.read_csv('NovosCandidatos.csv', sep = ';')
base_previsoes

# Mudança dos dados para formato de matriz
despesas = base_previsoes.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)

# Previsões e geração de nova base de dados com os valores originais e as previsões (0 não eleito; 1 eleito)
previsoes_teste = modelo.predict(despesas)
previsoes_teste

# Tabela completa de previsões para os novos candidatos
base_previsoes = np.column_stack((base_previsoes, previsoes_teste))
base_previsoes
```
#### [Voltar ao Sumário](#sumário)

## 8. Séries Temporais
Dados: "AirPassengers.csv"

Faça uma análise inicial dessa série temporal, decomponha e realize previsões com ARIMA.

#### 8.1. Tratamento de séries temporais
```
# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
#registro de converters para uso do matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Carregando a base de dados
base = pd.read_csv('AirPassengers.csv')
base.head()

# Visualizando o tipo de dados dos atributos
print(base.dtypes)

# Convertendo os atributos que estão no formato string para formato de data: ANO-MÊS
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
base = pd.read_csv('AirPassengers.csv', parse_dates = ['Month'],
                   index_col = 'Month', date_parser = dateparse)
base

# Visualizando o índice do dataframe (#Passengers)
base.index

# Criando a série temporal (ts)
ts = base['#Passengers']
ts

# Visualização de registro específico
ts[1]

# Visualização por ano e mês
ts['1949-02']

# Visualização de data específica
ts[datetime(1949,2,1)]

# Visualização de intervalos
ts['1950-01-01':'1950-07-31']

# Visualização de intervalos sem preencher a data de início
ts[:'1950-07-31']

# Visualização por ano
ts['1950']

# Valores máximos
ts.index.max()

# Valores mínimos
ts.index.min()

# Visualização da série temporal completa
plt.plot(ts)

# Visualização por ano (agregando a série temporal em intervalos anuais)
ts_ano = ts.resample('A').sum()
plt.plot(ts_ano)
ts_ano

# Visualização por mês
ts_mes = ts.groupby([lambda x: x.month]).sum()
plt.plot(ts_mes)

# Visualização entre datas específicas
ts_datas = ts['1960-01-01':'1960-12-01']
plt.plot(ts_datas)
```

#### 8.2. Decomposição
```
# Importando as bibliotecas
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
#registro de converters para uso do matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Carregando a base de dados, convertendo o atributo para data e criando a série temporal (ts)
base = pd.read_csv('AirPassengers.csv')
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
base = pd.read_csv('AirPassengers.csv', parse_dates = ['Month'],
                   index_col = 'Month', date_parser = dateparse)
ts = base['#Passengers']

# Visualizando a série temporal
plt.plot(ts)

# Decomposição da série temporal, criando uma variável para cada formato
decomposicao = seasonal_decompose(ts)

# Tendência
tendencia = decomposicao.trend
tendencia

# Sazonalidade
sazonal = decomposicao.seasonal
sazonal

# Erro
aleatorio = decomposicao.resid
aleatorio

# Gráfico para cada formato da série temporal
plt.plot(sazonal)

plt.plot(tendencia)

plt.plot(aleatorio)

# Todos os gráficos
plt.subplot(4,1,1)
plt.plot(ts, label = 'Original')
plt.legend(loc = 'best')

# Visualização somente da tendência
plt.subplot(4,1,2)
plt.plot(tendencia, label = 'Tendência')
plt.legend(loc = 'best')

# Visualização somente da sazonalidade
plt.subplot(4,1,3)
plt.plot(sazonal, label = 'Sazonalidade')
plt.legend(loc = 'best')

# Visualização somente do elemento aleatório
plt.subplot(4,1,4)
plt.plot(aleatorio, label = 'Aleatório')
plt.legend(loc = 'best')
plt.tight_layout()
```

#### 8.3. Previsões com Arima (AutoRegressive Integrated Moving Average)
```
# Importando as bibliotecas (inclusive instalando a pmdarima)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime

!pip install pmdarima
from pmdarima.arima import auto_arima

# Convertendo os atributos que estão no formato string para o formato ano-mês
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

plt.plot(data)

# O código abaixo usa a função auto_arima da biblioteca pmdarima para ajustar
# um modelo ARIMA aos dados fornecidos na variável data.
stepwise_model = auto_arima(data, start_p=1,start_q=1,start_d= 0, start_P=0, max_p=6, max_q=6, m=12, seasonal=True, trace=True, stepwise=False)

# Critério de Informação de Akaike (AIC, na sigla em inglês), que é uma medida usada para comparar modelos estatísticos
print(stepwise_model.aic())

# Separando treino e teste
train = data.loc['1949-01-01':'1959-12-01']
test = data.loc['1960-01-01':]

#Treino e previsão 12 meses para frente
stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=12)
future_forecast

# Comparando o que aconteceu de fato com os resultados previstos
future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["#Passengers"])
pd.concat([test,future_forecast],axis=1).plot() #azul test; laranja forecast

# Toda a série temporal com a previsão
pd.concat([data,future_forecast],axis=1).plot(linewidth=3)
```
#### [Voltar ao Sumário](#sumário)

## 9. Machine Learning
Dados: "Credit.csv"; "Credit2.csv"; "NovoCredit.csv"

Crie modelos de ML para indicar se os clientes são bons ('good') ou mau ('bad') pagadores.

#### 9.1. Naive Bayes
```
# Importando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

# Carregamento da base de dados e definição dos previsores (variáveis independentes - X) e a classe (variável dependente - y)
credito = pd.read_csv('Credit.csv')
credito.shape

credito.head()

# Formato de matriz
previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

# Transformação dos atributos categóricos em atributos numéricos, passando o índice de cada coluna categórica
# Necessário criar um objeto para cada atributo categórico, pois na sequência vai ser executado o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente, o que deixará o teste inconsistente
labelencoder1 = LabelEncoder()
previsores[:,0] = labelencoder1.fit_transform(previsores[:,0])

labelencoder2 = LabelEncoder()
previsores[:,2] = labelencoder2.fit_transform(previsores[:,2])

labelencoder3 = LabelEncoder()
previsores[:, 3] = labelencoder3.fit_transform(previsores[:, 3])

labelencoder4 = LabelEncoder()
previsores[:, 5] = labelencoder4.fit_transform(previsores[:, 5])

labelencoder5 = LabelEncoder()
previsores[:, 6] = labelencoder5.fit_transform(previsores[:, 6])

labelencoder6 = LabelEncoder()
previsores[:, 8] = labelencoder6.fit_transform(previsores[:, 8])

labelencoder7 = LabelEncoder()
previsores[:, 9] = labelencoder7.fit_transform(previsores[:, 9])

labelencoder8 = LabelEncoder()
previsores[:, 11] = labelencoder8.fit_transform(previsores[:, 11])

labelencoder9 = LabelEncoder()
previsores[:, 13] = labelencoder9.fit_transform(previsores[:, 13])

labelencoder10 = LabelEncoder()
previsores[:, 14] = labelencoder10.fit_transform(previsores[:, 14])

labelencoder11 = LabelEncoder()
previsores[:, 16] = labelencoder11.fit_transform(previsores[:, 16])

labelencoder12 = LabelEncoder()
previsores[:, 18] = labelencoder12.fit_transform(previsores[:, 18])

labelencoder13 = LabelEncoder()
previsores[:, 19] = labelencoder13.fit_transform(previsores[:, 19])

# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
X_teste

# Criação e treinamento do modelo (geração da tabela de probabilidades)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

# Previsões utilizando os registros de teste
previsoes = naive_bayes.predict(X_teste)
previsoes

# Geração da matriz de confusão e cálculo da taxa de acerto e erro
confusao = confusion_matrix(y_teste, previsoes)
confusao

taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
taxa_acerto

# Visualização da matriz de confusão
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

# Previsão com novo registro, transformando os atributos categóricos em numéricos
novo_credito = pd.read_csv('NovoCredit.csv')
novo_credito.shape

# Usando o mesmo objeto que foi criado antes, para manter o padrão dos dados
# Chama-se somente o método "transform", pois a adaptação aos dados (fit) já foi feita anteriormente
novo_credito = novo_credito.iloc[:,0:20].values
novo_credito[:,0] = labelencoder1.transform(novo_credito[:,0])
novo_credito[:, 2] = labelencoder2.transform(novo_credito[:, 2])
novo_credito[:, 3] = labelencoder3.transform(novo_credito[:, 3])
novo_credito[:, 5] = labelencoder4.transform(novo_credito[:, 5])
novo_credito[:, 6] = labelencoder5.transform(novo_credito[:, 6])
novo_credito[:, 8] = labelencoder6.transform(novo_credito[:, 8])
novo_credito[:, 9] = labelencoder7.transform(novo_credito[:, 9])
novo_credito[:, 11] = labelencoder8.transform(novo_credito[:, 11])
novo_credito[:, 13] = labelencoder9.transform(novo_credito[:, 13])
novo_credito[:, 14] = labelencoder10.transform(novo_credito[:, 14])
novo_credito[:, 16] = labelencoder11.transform(novo_credito[:, 16])
novo_credito[:, 18] = labelencoder12.transform(novo_credito[:, 18])
novo_credito[:, 19] = labelencoder13.transform(novo_credito[:, 19])

# Resultado da previsão
naive_bayes.predict(novo_credito)
```

#### 9.2. Árvores de Decisão
```
# Importando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

# Carregamento da base de dados e definição dos previsores (variáveis independentes - X) e classe (variável dependente - y)
credito = pd.read_csv('Credit.csv')
credito.shape

previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

# Conversão dos atributos categóricos para atributos numéricos, passando o índice de cada atributo categórico
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder.fit_transform(previsores[:, 6])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder.fit_transform(previsores[:, 9])
previsores[:, 11] = labelencoder.fit_transform(previsores[:, 11])
previsores[:, 13] = labelencoder.fit_transform(previsores[:, 13])
previsores[:, 14] = labelencoder.fit_transform(previsores[:, 14])
previsores[:, 16] = labelencoder.fit_transform(previsores[:, 16])
previsores[:, 18] = labelencoder.fit_transform(previsores[:, 18])
previsores[:, 19] = labelencoder.fit_transform(previsores[:, 19])

# Código otimizado utilizando um loop para aplicar a transformação com LabelEncoder
# Índices das colunas categóricas
# colunas_categoricas = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
# Aplicando LabelEncoder para as colunas categóricas
# labelencoder = LabelEncoder()
# for coluna in colunas_categoricas:
#    previsores[:, coluna] = labelencoder.fit_transform(previsores[:, coluna])

# Divisão da base de dados entre treinamento e teste. Usando 30% para testar e 70% para treinar.
# Random_state = 0 para sempre obter a mesma divisão da base quando o código for executado
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# Criação e treinamento do modelo
arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

# Exportação da árvore de decisão para o formato .dot, para posterior visualização
export_graphviz(arvore, out_file = 'tree.dot')

# Obtenção das previsões
previsoes = arvore.predict(X_teste)
previsoes

#matriz de confusão
confusao = confusion_matrix(y_teste, previsoes)
confusao

#taxa acerto
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_acerto

#taxa erro
taxa_erro = 1 - taxa_acerto
taxa_erro
```

#### 9.3. Seleção de Atributos

#### 9.4. Ensamble Learning com Random Forest

#### 9.5. Agrupamento com K-Means

#### 9.6. Agrupamento com C-Means



#### [Voltar ao Sumário](#sumário)

## 10. Neural Networks e Deep Learning
#### [Voltar ao Sumário](#sumário)

## 11. Grafos
#### [Voltar ao Sumário](#sumário)

## 12. Processamento de Linguagem Natural e Mineração de Texto
#### [Voltar ao Sumário](#sumário)

## 13. Bancos de Dados e Linguagem SQL
#### [Voltar ao Sumário](#sumário)

## 14. Bancos de Dados NoSQL e MongoDB
#### [Voltar ao Sumário](#sumário)

## 15. Computação na Nuvem
#### [Voltar ao Sumário](#sumário)

## 16. Spark com Databricks
#### [Voltar ao Sumário](#sumário)

</div>

