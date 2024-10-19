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

#### 1. Amostragem simples
Dados: iris.csv

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

#### 2. Medidas de Centralidade e Variabilidade
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

#### 3. Distribuição Normal
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

#### 1. Distribuição T de Student
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

#### 2. Distribuição de Poisson
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

#### 3. Distribuição Binomial
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

#### 4. Qui-Quadrado
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
Dados: slr12.csv

1. Franquias
- FraAnual: Taxa Anual
- CusInic: Investimento Inicial

2. Criar um modelo de regressão linear para prever qual será o Investimento inicial necessário de uma franquia a partir da Taxa Anual cobrado pelo franqueador.

```
#RESOLUÇÃO

# Importando as bibliotecas, sklearn para criar o modelo de regressão e yellowbrick para visualização de residuais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregando a base de dados
base = pd.read_csv('slr12.csv', sep=';')

# Definindo as variáveis X e y, X FraAnual é a variável independente e y CusInic é a variável dependente
X = base.iloc[:, 0].values
y = base.iloc[:, 1].values

# Calculando a correlação entre X e y
correlacao = np.corrcoef(X, y)
correlacao

#formato de matriz com uma coluna a mais
X = X.reshape(-1, 1)

# Criação do modelo e treinamento (fit indica que o treinamento deve ser executado)
modelo = LinearRegression()
modelo.fit(X, y)

# Gerando o gráfico com os pontos reais e as previsões
plt.scatter(X, y)
plt.plot(X, modelo.predict(X), color = 'red')

#valor anual da franquina
valr =  1300
modelo.predict([[valr]])
```

#### [Voltar ao Sumário](#sumário)

## 7. Regressão Logística
#### [Voltar ao Sumário](#sumário)

## 8. Séries Temporais
#### [Voltar ao Sumário](#sumário)

## 9. Machine Learning
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

