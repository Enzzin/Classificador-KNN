import sys
import numpy as np
from collections import Counter
import concurrent.futures # Biblioteca para processamento paralelo

# calc a dist euclidiana entre 2 vetores
def distancia_euclidiana(linha1, linha2):
    return np.sqrt(np.sum((linha1 - linha2)**2))

# acha os vizinhos mais proximos p/ uma amostra
def obter_vizinhos(conjunto_treino, amostra_teste, num_vizinhos):
    distancias = []
    for amostra_treino in conjunto_treino:
        dist = distancia_euclidiana(amostra_teste[1:], amostra_treino[1:])
        distancias.append((amostra_treino, dist))
    
    distancias.sort(key=lambda tup: tup[1])
    
    vizinhos = []
    for i in range(num_vizinhos):
        vizinhos.append(distancias[i][0])
    return vizinhos

# preve a classe de uma amostra com KNN
def prever_classificacao(conjunto_treino, amostra_teste, num_vizinhos):
    vizinhos = obter_vizinhos(conjunto_treino, amostra_teste, num_vizinhos)
    rotulos_vizinhos = [linha[0] for linha in vizinhos]
    previsao = Counter(rotulos_vizinhos).most_common(1)[0][0]
    return previsao

# carrega o arq de dados e converte p/ numeros
def carregar_dados(nome_arquivo):
    dados = []
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        for linha in linhas:
            linha_limpa = linha.strip().split()
            if linha_limpa:
                dados.append([float(x) for x in linha_limpa])
    return np.array(dados)

# calc a acuracia (reais vs previstos)
def calcular_acuracia(reais, previstos):
    acertos = 0
    for i in range(len(reais)):
        if reais[i] == previstos[i]:
            acertos += 1
    return (acertos / float(len(reais))) * 100.0

# Esta função agora executa o trabalho para UMA semente.
# Ela será chamada em paralelo para cada semente.
def testar_semente(semente, dados_originais, perc_treino, k):
    # cria uma cópia para não alterar o original
    dados = np.copy(dados_originais)
    
    # define a semente para esta execução
    np.random.seed(semente)
    np.random.shuffle(dados)

    # divide entre treino e teste
    tam_treino = int(len(dados) * (perc_treino / 100))
    conjunto_treino = dados[:tam_treino]
    conjunto_teste = dados[tam_treino:]

    previsoes = []
    rotulos_reais = [linha[0] for linha in conjunto_teste]
    
    for linha in conjunto_teste:
        previsao = prever_classificacao(conjunto_treino, linha, k)
        previsoes.append(previsao)

    acuracia = calcular_acuracia(rotulos_reais, previsoes)
    
    # Imprime o progresso (pode aparecer fora de ordem devido ao paralelismo)
    print(f"Semente: {semente:3d} testada, Acurácia: {acuracia:.2f}%")
    
    # Retorna a semente e sua acurácia
    return semente, acuracia

# func principal, q organiza tudo
def main():
    if len(sys.argv) != 5:
        print("Uso: python3 knn_busca_paralela.py <perc_treino> <perc_teste> <arquivo> <tam_vetor>")
        sys.exit(1)

    perc_treino = int(sys.argv[1])
    nome_arquivo = sys.argv[3]
    
    dados_originais = carregar_dados(nome_arquivo)
    
    melhor_acuracia = 0.0
    melhor_semente = -1
    num_seeds_para_testar = 200
    k = 3
    
    print(f"Iniciando busca paralela com {num_seeds_para_testar} sementes...")

    resultados = []
    # O 'with' garante que os processos sejam finalizados corretamente
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Cria uma lista de "tarefas futuras", uma para cada semente
        futures = [executor.submit(testar_semente, s, dados_originais, perc_treino, k) for s in range(num_seeds_para_testar)]
        
        # Coleta os resultados conforme eles ficam prontos
        for f in concurrent.futures.as_completed(futures):
            resultados.append(f.result())

    # Agora, com todos os resultados, encontra o melhor
    for semente, acuracia in resultados:
        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_semente = semente

    print("\n--- Resultado Final ---")
    print(f"Busca paralela finalizada após testar {num_seeds_para_testar} sementes.")
    print(f"A melhor semente encontrada foi a de número: {melhor_semente}")
    print(f"A maior acurácia alcançada foi de: {melhor_acuracia:.2f}%")

if __name__ == "__main__":
    main()