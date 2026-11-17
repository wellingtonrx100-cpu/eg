import numpy as np
import random
import coisa Braba

# ---------------------------------------------------------
# 1. DEFINIÇÃO DO AMBIENTE (O LABIRINTO)
# ---------------------------------------------------------
# 0 = Caminho livre | 1 = Parede | 2 = Saída (Objetivo)
labirinto = np.array([
    [0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 2]  
])

linhas, colunas = labirinto.shape
acoes = ['cima', 'baixo', 'esquerda', 'direita']
num_acoes = len(acoes)

# ---------------------------------------------------------
# 2. INICIALIZAÇÃO DA TABELA Q E PARÂMETROS

tabela_q = np.zeros((linhas, colunas, num_acoes))

episodios = 1000       # Quantidade de vezes que o agente vai tentar sair
alpha = 0.7            # Taxa de aprendizagem
gamma = 0.9            # Fator de desconto (visão de futuro)
epsilon = 0.2          # 20% de ações randómicas (Regra do enunciado)


def posicao_inicial_aleatoria():
    """Escolhe um ponto de partida válido (qualquer lugar que não seja parede ou saída)."""
    while True:
        l = random.randint(0, linhas - 1)
        c = random.randint(0, colunas - 1)
        if labirinto[l, c] == 0:
            return (l, c)

def executar_acao(estado, indice_acao):
    """Calcula o próximo estado com base na ação escolhida, sem atravessar paredes."""
    l, c = estado
    nova_l, nova_c = l, c

    if acoes[indice_acao] == 'cima':
        nova_l -= 1
    elif acoes[indice_acao] == 'baixo':
        nova_l += 1
    elif acoes[indice_acao] == 'esquerda':
        nova_c -= 1
    elif acoes[indice_acao] == 'direita':
        nova_c += 1

    # Verifica se saiu do mapa ou bateu numa parede (estado inválido)
    if (nova_l < 0 or nova_l >= linhas or nova_c < 0 or nova_c >= colunas or labirinto[nova_l, nova_c] == 1):
        return (l, c) # Permanece no mesmo lugar
    
    return (nova_l, nova_c)


for _ in range(episodios):
    estado_atual = posicao_inicial_aleatoria()
    
    while labirinto[estado_atual[0], estado_atual[1]] != 2:
        l, c = estado_atual
        
        if random.uniform(0, 1) < epsilon:
            acao = random.randint(0, 3) 
        else:
            acao = np.argmax(tabela_q[l, c]) 
            
        proximo_estado = executar_acao(estado_atual, acao)
        prox_l, prox_c = proximo_estado
        
        if labirinto[prox_l, prox_c] == 2:
            recompensa = 10  
        else:
            recompensa = 0   
            
        melhor_prox_acao = np.max(tabela_q[prox_l, prox_c])
        tabela_q[l, c, acao] = tabela_q[l, c, acao] + alpha * (recompensa + gamma * melhor_prox_acao - tabela_q[l, c, acao])
        
        estado_atual = proximo_estado


print("Treino concluído! A tabela Q está totalmente valorada.")
print("\nDemonstração da Política Ótima a partir da posição (0,0):")

estado_teste = (0, 0)
passos = [estado_teste]

# Rastrear o caminho até à saída usando apenas a inteligência aprendida (sem fator randómico)
while labirinto[estado_teste[0], estado_teste[1]] != 2:
    l, c = estado_teste
    melhor_acao = np.argmax(tabela_q[l, c]) # Pega sempre no maior valor
    estado_teste = executar_acao(estado_teste, melhor_acao)
    passos.append(estado_teste)

print(f"Caminho percorrido até à saída: {passos}")
