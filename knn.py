import sys
import numpy as np
import random
from collections import Counter

# calc a dist euclidiana entre 2 vetores
def distancia_euclidiana(linha1, linha2):
    return np.sqrt(np.sum((linha1 - linha2)**2))

# acha os vizinhos mais proximos p/ uma amostra
def obter_vizinhos(conjunto_treino, amostra_teste, num_vizinhos):
    distancias = []
    for amostra_treino in conjunto_treino:
        # a 1a coluna eh o rotulo, o resto sao os atributos
        dist = distancia_euclidiana(amostra_teste[1:], amostra_treino[1:])
        distancias.append((amostra_treino, dist))
    
    # ordena pela distancia
    distancias.sort(key=lambda tup: tup[1])
    
    vizinhos = []
    for i in range(num_vizinhos):
        vizinhos.append(distancias[i][0])
    return vizinhos

# preve a classe de uma amostra com KNN
def prever_classificacao(conjunto_treino, amostra_teste, num_vizinhos):
    vizinhos = obter_vizinhos(conjunto_treino, amostra_teste, num_vizinhos)
    # pega so os rotulos dos vizinhos
    rotulos_vizinhos = [linha[0] for linha in vizinhos]
    # retorna o rotulo mais comum (voto)
    previsao = Counter(rotulos_vizinhos).most_common(1)[0][0]
    return previsao

# carrega o arq de dados e converte p/ numeros
def carregar_dados(nome_arquivo):
    dados = []
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            # limpa e divide a linha
            linha_limpa = linha.strip().split()
            if linha_limpa:
                # converte tudo p/ numero e joga na lista de dados
                dados.append([float(x) for x in linha_limpa])
    return np.array(dados)

# calc a acuracia (reais vs previstos)
def calcular_acuracia(reais, previstos):
    acertos = 0
    for i in range(len(reais)):
        if reais[i] == previstos[i]:
            acertos += 1
    return (acertos / float(len(reais))) * 100.0

# gera a matriz de confusao
def gerar_matriz_confusao(reais, previstos, num_classes=10):
    # assume classes de 0 a 9
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(reais)):
        # converte p/ int p/ usar como indice
        idx_real = int(reais[i])
        idx_previsto = int(previstos[i])
        matriz[idx_real][idx_previsto] += 1
    return matriz


def main():
    # checa se os argumentos estao certos
    if len(sys.argv) != 5:
        sys.exit(1)

    # le os parametros da linha de comando
    try:
        perc_treino = int(sys.argv[1])
        nome_arquivo = sys.argv[3]
    except (ValueError, IndexError):
        print("Erro: Parametros invalidos")
        sys.exit(1)


    #seed 
    # 80 20 87.50%  seed:25
    # 90 10 91.50%  seed:112
    #np.random.seed(112)
    # carrega e prepara os dados
    dados = carregar_dados(nome_arquivo)
    np.random.shuffle(dados) # embaralha os dados

    # divide entre treino e teste
    tam_treino = int(len(dados) * (perc_treino / 100))
    conjunto_treino = dados[:tam_treino]
    conjunto_teste = dados[tam_treino:]

    print(f"Total de amostras: {len(dados)}")
    print(f"Tamanho do conjunto de treino: {len(conjunto_treino)}")
    print(f"Tamanho do conjunto de teste: {len(conjunto_teste)}")
    
    # o valor de K eh fixo em 3 (valor q teve a melhor acuracia)
    k = 3
    
    # faz a previsao p/ cada amostra de teste
    previsoes = []
    rotulos_reais = [linha[0] for linha in conjunto_teste]
    
    for linha in conjunto_teste:
        previsao = prever_classificacao(conjunto_treino, linha, k)
        previsoes.append(previsao)

    # usa funcs p/ calc acuracia e matriz
    acuracia = calcular_acuracia(rotulos_reais, previsoes)
    matriz_confusao = gerar_matriz_confusao(rotulos_reais, previsoes)
    
    print(f"\nAcuracia do classificador foi de {acuracia:.2f}%")
    
    print("\nMatriz de Confusao:")
    print(matriz_confusao)

if __name__ == "__main__":
    main()