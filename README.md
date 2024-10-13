<div align="justify">
  
# Compilado de exercícios práticos de Data Science utilizando Python
#### Autor: Guilherme Oliveira da Rocha Cunha

Estes exercícios foram apresentados no curso **Formação Cientista de Dados: O Curso Completo** oferecido pela plataforma de cursos online [Udemy](https://www.udemy.com/pt/). Este é um curso de Data Science em que se conhece e aprende a aplicar todos os principais conceitos e técnicas para se qualificar e atuar como um Cientista de Dados, com videos explicativos e detalhados, exemplos práticos de codificação em R e Python usando dados reais e explicações de resolução de fórmulas passo a passo. A resolução de todos os exercícios foi feita por mim, construída de acordo com as instruções do professor Fernando Amaral, instrutor responsável por este curso.

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
Tratar dados 'tempo.csv'. Faça cada etapa de forma separada e  lembre-se de tratar os valores NaNs.

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

#importando dados
dataset = pd.read_csv("tempo.csv", sep=";")
#visualizar
dataset.head()

#explorando dados categóricos
#Aparencia
agrupado_aparencia = dataset.groupby(['Aparencia']).size()
agrupado_aparencia.plot.bar(color = 'gray')

#Vento
agrupado_vento = dataset.groupby(['Vento']).size()
agrupado_vento.plot.bar(color = 'gray')

#Jogar
agrupado_jogar = dataset.groupby(['Jogar']).size()
agrupado_jogar.plot.bar(color = 'gray')

#explorando dados numéricos
#Temperatura
dataset['Temperatura'].describe()

#Umidade
dataset['Umidade'].describe()

#contando valores NAN
dataset.isnull().sum()

#substituindo o valor inválido "menos" pelo valor "sol"
dataset.loc[dataset['Aparencia'] ==  'menos', 'Aparencia'] = "sol"
#visualiza o resultado
agrupado = dataset.groupby(['Aparencia']).size()
agrupado

#visualizando os valores de "Temperatura" que estão fora do domínio
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130) ]

#calculando a mediana e substituindo o valor fora do domínio por ela
mediana = sts.median(dataset['Temperatura'])
dataset.loc[(dataset['Temperatura'] <  -130 )  | ( dataset['Temperatura'] >  130), 'Temperatura'] = mediana

#umidade, dominio e NaNs
agrupado = dataset.groupby(['Umidade']).size()
agrupado

#total de NaNs
dataset['Umidade'].isnull().sum()

#calculando a mediana e preenchendo os NaNs
mediana = sts.median(dataset['Umidade'])
dataset['Umidade'].fillna(mediana, inplace=True)

#total de NaNs para "Vento"
dataset['Vento'].isnull().sum()

#preenchendo com "FALSO" (que é a maior ocorrência)
dataset['Vento'].fillna('FALSO', inplace=True)
```
#### [Voltar ao Sumário](#sumário)

## 3. Gráficos, Visualização e Dashboards
#### [Voltar ao Sumário](#sumário)

## 4. Estatística I
#### [Voltar ao Sumário](#sumário)

## 5. Estatística II
#### [Voltar ao Sumário](#sumário)

## 6. Regressão Linear
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

