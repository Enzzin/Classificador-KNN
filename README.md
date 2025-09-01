# Classificador KNN e Busca Paralela por Semente Ótima

## 📝 Visão Geral do Projeto

Este repositório contém um trabalho acadêmico que explora o algoritmo de classificação **K-Nearest Neighbors (KNN)**. O projeto inclui duas implementações principais em Python:

1.  Um classificador KNN completo, implementado do zero (`knn.py`), que treina e avalia o modelo, exibindo a acurácia e a matriz de confusão.
2.  Um script de otimização (`seed.py`) que utiliza processamento paralelo para testar centenas de sementes de aleatoriedade, identificando qual delas produz a melhor acurácia na divisão dos dados.

O objetivo é não apenas demonstrar o funcionamento do KNN, mas também explorar como a inicialização do processo de embaralhamento de dados pode impactar o desempenho do modelo.

## 📂 Estrutura do Repositório

-   `knn.py`: Script principal com a implementação do classificador KNN.
-   `seed.py`: Script para busca paralela da melhor semente de aleatoriedade para maximizar a acurácia.
-   `outpux.txt`: O conjunto de dados (dataset) utilizado para treinar e testar os modelos.

## 📜 Descrição dos Arquivos

### `knn.py`
Este script é uma implementação pura do algoritmo KNN, sem o uso de bibliotecas de Machine Learning como Scikit-learn. Suas principais funcionalidades são:
-   **Carregamento de Dados:** Lê o arquivo de dados e o converte para uma estrutura NumPy.
-   **Divisão Treino/Teste:** Embaralha os dados e os divide em conjuntos de treino e teste com base em uma porcentagem definida pelo usuário.
-   **Cálculo de Distância:** Utiliza a distância euclidiana para medir a similaridade entre as amostras.
-   **Classificação:** Para cada amostra de teste, encontra os *k* vizinhos mais próximos no conjunto de treino e prevê a classe através de um sistema de "votação" majoritária.
-   **Avaliação:** Calcula e exibe a acurácia final do modelo e gera uma matriz de confusão para uma análise detalhada dos acertos e erros por classe.

### `seed.py`
Este script aborda o problema da variabilidade de resultados em Machine Learning. A forma como os dados são embaralhados (definida pela "semente" ou *seed*) antes da divisão pode levar a diferentes acurácias. O objetivo deste script é encontrar a **semente ótima**.
-   **Busca em Paralelo:** Utiliza a biblioteca `concurrent.futures` do Python para testar um grande número de sementes (neste caso, 200) de forma paralela, aproveitando múltiplos núcleos do processador e acelerando a busca.
-   **Teste por Semente:** Para cada semente, o script executa o fluxo completo do KNN (embaralhar, dividir, treinar, prever) e calcula a acurácia.
-   **Resultado Final:** Ao final, o script reporta qual semente resultou na maior acurácia, permitindo uma divisão de dados mais favorável para o modelo.

### `outpux.txt`
Este é o arquivo de dados. Cada linha representa uma amostra, onde:
-   A **primeira coluna** é o rótulo (a classe/categoria) da amostra.
-   As **colunas restantes** formam o vetor de características daquela amostra.

## 🚀 Como Executar

### Pré-requisitos
- Python 3
- Biblioteca NumPy

Você pode instalar o NumPy com o seguinte comando:
```bash
pip install numpy
```

### 1. Executando o Classificador KNN (`knn.py`)
Para rodar o classificador principal, use o seguinte comando no terminal:
```bash
python knn.py <perc_treino> <perc_teste> <arquivo> <tam_vetor>
```
-   `<perc_treino>`: Porcentagem de dados para treino (ex: 90).
-   `<perc_teste>`: Porcentagem de dados para teste (ex: 10).
-   `<arquivo>`: Nome do arquivo de dados (ex: `outpux.txt`).
-   `<tam_vetor>`: Argumento para o tamanho do vetor.

**Exemplo de uso:**
```bash
python knn.py 90 10 outpux.txt 25
```

### 2. Executando a Busca pela Melhor Semente (`seed.py`)
Para iniciar a busca paralela pela semente com melhor desempenho, use um comando similar:
```bash
python seed.py <perc_treino> <perc_teste> <arquivo> <tam_vetor>
```
**Exemplo de uso:**
```bash
python seed.py 90 10 outpux.txt 25
```
O script irá testar 200 sementes e, ao final, imprimirá a melhor semente encontrada e a acurácia correspondente.

## 🛠️ Tecnologias Utilizadas
- **Python 3**
- **NumPy** para cálculos numéricos e manipulação de vetores.
- **`concurrent.futures`** para a implementação do processamento paralelo.
