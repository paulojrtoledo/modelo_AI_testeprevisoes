# Projeto Python IA: Inteligência Artificial e Previsões - Score de Crédito dos Clientes
## Visão Geral

Este projeto em Python tem como objetivo desenvolver um modelo de Inteligência Artificial capaz de prever o score de crédito de clientes de um banco. Através da análise de dados históricos dos clientes, o modelo aprende a identificar padrões e, com base nas informações de novos clientes, classifica seu score de crédito como Ruim, Ok ou Bom.


Arquivos da aula: https://drive.google.com/drive/folders/1FbDqVq4XLvU85VBlVIMJ73p9oOu6u2-J?usp=drive_link


## Case

O desafio proposto por um banco foi analisar a base de dados de seus clientes para construir um modelo preditivo automático do score de crédito. Este modelo permitirá ao banco avaliar o risco de crédito de novos clientes de forma eficiente.

## Passo a Passo da Implementação

1.  **Entendimento do Desafio e da Empresa:** Compreensão do problema de negócio e do contexto bancário.
2.  **Importação da Base de Dados:** Utilização da biblioteca pandas para carregar e visualizar os dados dos clientes a partir do arquivo `clientes.csv`.
    ```python
    import pandas as pd

    tabela = pd.read_csv("clientes.csv")
    display(tabela)
    ```
3.  **Preparação da Base de Dados para IA:**
    * Análise das informações da tabela (`tabela.info()`) para entender os tipos de dados e identificar possíveis valores ausentes.
    * Utilização do `LabelEncoder` da biblioteca scikit-learn para converter colunas categóricas (`profissao`, `mix_credito`, `comportamento_pagamento`) em representações numéricas, necessárias para o treinamento dos modelos de machine learning.
    ```python
    from sklearn.preprocessing import LabelEncoder

    codificador1 = LabelEncoder()
    tabela["profissao"] = codificador1.fit_transform(tabela["profissao"])

    codificador2 = LabelEncoder()
    tabela["mix_credito"] = codificador2.fit_transform(tabela["mix_credito"])

    codificador3 = LabelEncoder()
    tabela["comportamento_pagamento"] = codificador3.fit_transform(tabela["comportamento_pagamento"])

    display(tabela.info())
    ```
4.  **Separação dos Dados:**
    * Definição da variável alvo (`y`, coluna `score_credito`) e das variáveis preditoras (`x`, todas as outras colunas exceto `score_credito` e `id_cliente`).
    * Divisão dos dados em conjuntos de treino (para o modelo aprender) e teste (para avaliar o desempenho do modelo em dados não vistos) utilizando a função `train_test_split` do scikit-learn.
    ```python
    y = tabela["score_credito"]
    x = tabela.drop(columns=["score_credito", "id_cliente"])

    from sklearn.model_selection import train_test_split
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
    ```
5.  **Criação e Treinamento dos Modelos de IA:**
    * Importação de dois modelos de classificação populares: `RandomForestClassifier` (um modelo de ensemble baseado em árvores de decisão) e `KNeighborsClassifier` (um modelo baseado na proximidade de vizinhos).
    * Instanciação e treinamento de ambos os modelos utilizando os dados de treino (`x_treino`, `y_treino`) através do método `.fit()`.
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    modelo_arvoredecisao = RandomForestClassifier()
    modelo_knn = KNeighborsClassifier()

    modelo_arvoredecisao.fit(x_treino, y_treino)
    modelo_knn.fit(x_treino, y_treino)
    ```
6.  **Avaliação dos Modelos:**
    * Realização de previsões nos dados de teste (`x_teste`) utilizando os modelos treinados com o método `.predict()`.
    * Cálculo da acurácia de cada modelo comparando as previsões com os valores reais de `y_teste` utilizando a função `accuracy_score` do scikit-learn.
    * Exibição das acurácias para comparar o desempenho dos modelos.
    ```python
    previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
    previsao_knn = modelo_knn.predict(x_teste)

    from sklearn.metrics import accuracy_score
    from IPython.display import display

    display(f"Acurácia Árvore de Decisão: {accuracy_score(y_teste, previsao_arvoredecisao):.1%}")
    display(f"Acurácia KNN: {accuracy_score(y_teste, previsao_knn):.1%}")
    ```
7.  **Escolha do Melhor Modelo:**
    * Com base na maior acurácia obtida nos dados de teste, o modelo de **Árvore de Decisão (Random Forest)** foi escolhido como o melhor modelo para este problema.
8.  **Realização de Novas Previsões:**
    * Carregamento de um novo conjunto de dados de clientes a partir do arquivo `novos_clientes.csv`.
    * Aplicação das mesmas transformações de codificação nas colunas categóricas da nova tabela, utilizando os mesmos objetos `LabelEncoder` ajustados anteriormente.
    * Utilização do modelo de Árvore de Decisão treinado para prever o score de crédito dos novos clientes com o método `.predict()`.
    * Exibição das previsões.
    ```python
    tabela_nova = pd.read_csv("novos_clientes.csv")
    display(tabela_nova)

    tabela_nova["profissao"] = codificador1.fit_transform(tabela_nova["profissao"])
    tabela_nova["mix_credito"] = codificador2.fit_transform(tabela_nova["mix_credito"])
    tabela_nova["comportamento_pagamento"] = codificador3.fit_transform(tabela_nova["comportamento_pagamento"])

    previsao = modelo_arvoredecisao.predict(tabela_nova)
    display(previsao)
    ```

## Arquivos

* `clientes.csv`: Arquivo contendo os dados históricos dos clientes para treinamento e teste do modelo.
* `novos_clientes.csv`: Arquivo contendo os dados de novos clientes para os quais o modelo fará a previsão do score de crédito.

## Bibliotecas Utilizadas

* `pandas`: Para manipulação e análise de dados tabulares.
* `scikit-learn`: Biblioteca de machine learning em Python, utilizada para pré-processamento de dados (`LabelEncoder`), divisão de dados (`train_test_split`), implementação dos modelos de classificação (`RandomForestClassifier`, `KNeighborsClassifier`) e avaliação de desempenho (`accuracy_score`).
* `IPython.display`: Para exibir informações de forma mais amigável, como a acurácia do modelo.

## Próximos Passos (Opcional)

* Análise mais aprofundada das métricas de avaliação do modelo (precisão, recall, F1-score, matriz de confusão).
* Otimização dos hiperparâmetros do modelo de Random Forest para tentar melhorar ainda mais o desempenho.
* Exploração de outros algoritmos de classificação.
* Implementação de técnicas de feature engineering para criar novas variáveis que possam melhorar a capacidade preditiva do modelo.
* Desenvolvimento de uma interface para utilizar o modelo em um ambiente de produção.

## 📌 Como usar este projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/paulojrtoledo/modelo_AI_testeprevisoes.git
