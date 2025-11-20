import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Nova importação para o heatmap
import os
import random
import copy
from collections import deque # Necessário para o Tabu Search
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set beautiful plot style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# --- 1. PREPARAÇÃO DOS DADOS ---

def get_problem_data(asset_filenames, base_path='archive/'):
    """
    Processa os ficheiros CSV para extrair os parâmetros do problema:
    p (preços), mu (retornos esperados), sigma (covariância).
    """
    n = len(asset_filenames)
    all_returns = []
    prices = []
    tickers = []
    
    base_path = 'archive/'
    
    for filename in asset_filenames:
        try:
            filepath = os.path.join(base_path, filename)
            ticker = filename.split('.')[0]
            
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            
            # Set Date as index to align correctly with other stocks
            df.set_index('Date', inplace=True)
            
            # Get last available price
            last_price = df['Close'].iloc[-1]
            
            # Calculate returns
            daily_returns = df['Close'].pct_change(fill_method=None)
            
            # Append only if we have data
            if not daily_returns.empty:
                all_returns.append(daily_returns)
                tickers.append(ticker)
                prices.append(last_price)
            
        except FileNotFoundError:
            print(f"Erro: Ficheiro não encontrado em {filepath}")
            continue # Skip this file and continue
        except Exception as e:
            print(f"Erro ao processar {filepath}: {e}")
            continue # Skip this file

    if not all_returns:
        print("Erro: Nenhum dado de retorno válido foi carregado.")
        return None, None, None, 0, []

    # Concatenate along dates (aligns by index)
    returns_df = pd.concat(all_returns, axis=1, keys=tickers)
    
    # --- THE FIX IS HERE ---
    # Old: returns_df = returns_df.dropna()  <- Deleted everything not matching dates
    # New: Fill missing dates with 0 (assumes flat return if stock not trading)
    returns_df = returns_df.fillna(0) 
    # -----------------------
    
    mu = returns_df.mean().values * 252
    sigma = returns_df.cov() * 252 
    
    p = np.array(prices)
    
    # Update n in case some files failed to load
    n = len(tickers)
    
    print("--- Dados de Problema Carregados ---")
    print(f"N (Número de ativos): {n}")
    
    # Gerar o heatmap da covariância
    if n > 1:
        plot_covariance_heatmap(sigma, tickers)
    
    sigma_np = sigma.values
    
    print(f"Sigma (Covariância Anualizada) shape: {sigma_np.shape}")
    print("------------------------------------")
    
    return p, mu, sigma_np, n, tickers

# --- 2. DEFINIÇÃO DO PROBLEMA (IA.pdf) ---

B = 5000.0   # Orçamento total
k = 1000.0   # Investimento máximo por ativo

# MUDANÇA CRÍTICA AQUI:
# Como o risco agora é medido em "Dólares ao quadrado", ele é um número gigante.
# Temos de baixar drasticamente o lambda para compensar.
lambda_val = 0.0001  # Antes era 0.1 (Reduzido 1000x)

# Penalidades (Soft Constraints)
# Devem ser altas o suficiente para doer, mas não para matar a exploração.
alpha = 10.0   # Penalidade de orçamento (por dólar violado)
beta = 10.0    # Penalidade de limite por ativo (por dólar violado)

def calculate_return(x, mu, p):
    # Expected Dollar Return = Sum(Shares * Price * Expected_Rate)
    return np.dot(x * p, mu) 

def calculate_risk(x, sigma, p):
    # Portfolio Dollar Variance = (Value_Vector).T * Covariance * (Value_Vector)
    # We use (x * p) to get the vector of dollars invested in each asset
    invested_vector = x * p
    return np.dot(invested_vector.T, np.dot(sigma, invested_vector))

def calculate_penalty(x, p, B, k, alpha, beta):
    total_investment = np.dot(p, x)
    penalty_B = alpha * max(0, total_investment - B)
    
    individual_investments = p * x
    violations_k = [max(0, inv - k) for inv in individual_investments]
    penalty_k = beta * sum(violations_k)
    
    return penalty_B + penalty_k

def calculate_fitness(x, mu, sigma, p, B, k, lambda_val, alpha, beta):
    x = np.maximum(0, x)
    
    # Pass 'p' to the new logic
    retorno = calculate_return(x, mu, p) 
    risco = calculate_risk(x, sigma, p)
    
    utility = retorno - lambda_val * risco
    penalty = calculate_penalty(x, p, B, k, alpha, beta)
    return utility - penalty

# --- 3. ALGORITMO 1: HILL CLIMBING ---

def hill_climbing(n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=1000):
    # Inicialização: Usa a lógica do GA para criar uma solução inicial válida e não vazia
    current_x = ga_create_individual(n, p, B, k)
    current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
    history = [current_fitness]
    
    target_step_value = k * 0.10  # Passo de ~$100
    
    for _ in range(max_iter):
        best_neighbor_x = current_x
        best_neighbor_fitness = current_fitness
        
        # Tenta melhorar cada ativo
        for j in range(n):
            if p[j] <= 0: continue
            
            # Passo dinâmico: quantas ações valem ~$100?
            step = max(1, int(target_step_value / p[j]))
            
            # --- Movimento 1: Adicionar ---
            # Verifica se adicionar viola grosseiramente os limites antes de calcular fitness
            # (Otimização de performance simples)
            if (current_x[j] + step) * p[j] <= k * 1.1: # Permite pequena violação para a penalidade tratar
                x_plus = np.copy(current_x)
                x_plus[j] += step
                fit_plus = calculate_fitness(x_plus, mu, sigma, p, B, k, lambda_val, alpha, beta)
                
                if fit_plus > best_neighbor_fitness:
                    best_neighbor_x = x_plus
                    best_neighbor_fitness = fit_plus
            
            # --- Movimento 2: Remover ---
            if current_x[j] >= step:
                x_minus = np.copy(current_x)
                x_minus[j] -= step
                fit_minus = calculate_fitness(x_minus, mu, sigma, p, B, k, lambda_val, alpha, beta)
                
                if fit_minus > best_neighbor_fitness:
                    best_neighbor_x = x_minus
                    best_neighbor_fitness = fit_minus

        if best_neighbor_fitness <= current_fitness:
            break
            
        current_x = best_neighbor_x
        current_fitness = best_neighbor_fitness
        history.append(current_fitness)
        
    return current_x, current_fitness, history

# --- 4. ALGORITMO 2: SIMULATED ANNEALING ---

def get_random_neighbor(x, n, p, B, k, aggressive=True):
    """
    Gera vizinho com passos dinâmicos baseados no valor monetário.
    """
    x_neighbor = np.copy(x)
    
    # Tenta modificar um valor monetário fixo (ex: $100 ou 10% do limite k)
    # Isto evita passos de formiga em ações baratas
    target_step_value = k * 0.10  # Passo de ~$100
    
    # Escolhe operação
    current_investment = np.dot(p, x_neighbor)
    budget_available = B - current_investment
    
    # Probabilidade adaptativa
    if budget_available > target_step_value: 
        operation = 'add' if random.random() < 0.7 else 'remove'
    elif budget_available > 0:
        operation = 'add' if random.random() < 0.5 else 'remove'
    else:
        operation = 'add' if random.random() < 0.2 else 'remove'
    
    asset_idx = random.randint(0, n - 1)
    price = p[asset_idx]
    
    if price <= 0: return x_neighbor # Proteção contra preço zero
    
    # Calcula quantas ações correspondem ao target_step_value
    # Ex: Se target=$100 e preço=$2, step_size=50 ações. Se preço=$100, step_size=1 ação.
    dynamic_step = max(1, int(target_step_value / price))
    
    if operation == 'add':
        # Verifica quanto ainda cabe no limite individual (k) e no global (B)
        current_val = x_neighbor[asset_idx] * price
        space_in_k = k - current_val
        space_in_B = budget_available
        
        limit_monetary = min(space_in_k, space_in_B)
        
        if limit_monetary > 0:
            # Quantas ações cabem nesse dinheiro?
            max_shares_possible = int(limit_monetary / price)
            if max_shares_possible > 0:
                # Adiciona entre 1 e o passo dinâmico (mas não mais que o possível)
                limit = min(dynamic_step, max_shares_possible)
                add_amount = random.randint(1, max(1, limit))
                x_neighbor[asset_idx] += add_amount
                
    else: # remove
        if x_neighbor[asset_idx] > 0:
            # Remove até ao passo dinâmico, mas não mais do que tem
            limit = min(dynamic_step, x_neighbor[asset_idx])
            remove_amount = random.randint(1, max(1, limit))
            x_neighbor[asset_idx] -= remove_amount

    return x_neighbor

def simulated_annealing(n, mu, sigma, p, B, k, lambda_val, alpha, beta, 
                        max_iter=10000, T_initial=1000.0, T_final=0.1, 
                        cooling_rate=0.99, aggressive_neighbors=True):
    """
    Simulated Annealing com opção de estratégia de vizinhança.
    
    aggressive_neighbors=True: Usa mais budget rapidamente (pode sair da fronteira)
    aggressive_neighbors=False: Exploração mais conservadora (melhor para Pareto)
    """
    current_x = np.zeros(n, dtype=int)
    current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
    best_x = current_x
    best_fitness = current_fitness
    T = T_initial
    history = []
    
    for i in range(max_iter):
        if T <= T_final: break
        new_x = get_random_neighbor(current_x, n, p, B, k, aggressive=aggressive_neighbors)
        new_fitness = calculate_fitness(new_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
        delta_fitness = new_fitness - current_fitness
        
        if delta_fitness > 0:
            current_x = new_x; current_fitness = new_fitness
            if new_fitness > best_fitness:
                best_x = new_x; best_fitness = new_fitness
        elif random.random() < np.exp(delta_fitness / T):
            current_x = new_x; current_fitness = new_fitness
                
        T *= cooling_rate
        history.append(best_fitness)
        
    return best_x, best_fitness, history

# --- 5. ALGORITMO 3: GENETIC ALGORITHM ---

def ga_create_individual(n, p, B, k):
    """ Cria indivíduo considerando o preço das ações """
    individual = np.zeros(n, dtype=int)
    for i in range(n):
        if p[i] > 0:
            # Calcula quantas ações cabem no limite k ($1000)
            max_shares = int(k / p[i])
            # Inicializa aleatoriamente entre 0 e 50% do limite para não estourar o budget logo de início
            if max_shares > 0:
                individual[i] = random.randint(0, int(max_shares * 0.5))
    return individual

def ga_selection(population, fitnesses):
    tournament_size = 3
    indices = np.random.randint(0, len(population), tournament_size)
    best_index = indices[np.argmax(fitnesses[indices])]
    return population[best_index]

def ga_crossover(parent1, parent2):
    n = len(parent1)
    child = np.zeros(n, dtype=int)
    for i in range(n):
        child[i] = parent1[i] if random.random() < 0.5 else parent2[i]
    return child

def ga_mutation(individual, n, p, k, mutation_rate=0.05):
    if random.random() < mutation_rate:
        index = random.randint(0, n - 1)
        # Passo de mutação dinâmico: ~10% do valor permitido
        step = max(1, int((k / p[index]) * 0.1)) if p[index] > 0 else 1
        
        change = random.randint(1, step)
        if random.random() < 0.5: individual[index] += change
        else: individual[index] -= change
        
        individual[index] = max(0, individual[index])
    return individual

def genetic_algorithm(n, mu, sigma, p, B, k, lambda_val, alpha, beta,
                      pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.1):
    
    # FIX: Passar p, B, k para a criação
    population = [ga_create_individual(n, p, B, k) for _ in range(pop_size)]
    fitnesses = np.array([calculate_fitness(ind, mu, sigma, p, B, k, lambda_val, alpha, beta) for ind in population])
    
    best_x = population[np.argmax(fitnesses)]
    best_fitness = np.max(fitnesses)
    history = [best_fitness]
    
    for gen in range(generations):
        new_population = []
        elite_index = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(population[elite_index]))
        
        while len(new_population) < pop_size:
            parent1 = ga_selection(population, fitnesses)
            parent2 = ga_selection(population, fitnesses)
            
            child = ga_crossover(parent1, parent2) if random.random() < crossover_rate else copy.deepcopy(parent1)
            # FIX: Passar p e k para mutação
            child = ga_mutation(child, n, p, k, mutation_rate)
            new_population.append(child)
            
        population = new_population
        fitnesses = np.array([calculate_fitness(ind, mu, sigma, p, B, k, lambda_val, alpha, beta) for ind in population])
        
        current_best_fitness = np.max(fitnesses)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_x = population[np.argmax(fitnesses)]
            
        history.append(best_fitness)
        
        if (gen + 1) % 50 == 0:
            print(f"  GA Geração {gen+1}/{generations} - Melhor Fitness: {best_fitness:.4f}")

    return best_x, best_fitness, history

# --- 6. ALGORITMO 4: TABU SEARCH (NOVO) ---

def get_neighbors_ts(x, n, n_neighbors, p, B, k):
    """ Gera um conjunto de vizinhos para o Tabu Search """
    neighbors = []
    for _ in range(n_neighbors):
        neighbors.append(get_random_neighbor(x, n, p, B, k, aggressive=True))
    return neighbors

def tabu_search(n, mu, sigma, p, B, k, lambda_val, alpha, beta,
                max_iter=1000, n_neighbors=50, tabu_tenure=20):
    
    current_x = np.zeros(n, dtype=int)
    current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
    
    best_x = current_x
    best_fitness = current_fitness
    history = [best_fitness]
    
    # tabu_list armazena 'tuplos' das soluções para serem "hashable"
    tabu_list = deque(maxlen=tabu_tenure) 
    
    for i in range(max_iter):
        neighbors = get_neighbors_ts(current_x, n, n_neighbors, p, B, k)
        
        best_neighbor_x = None
        best_neighbor_fitness = -np.inf
        
        for neighbor_x in neighbors:
            neighbor_tuple = tuple(neighbor_x)
            neighbor_fitness = calculate_fitness(neighbor_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
            
            # Critério de Aspiração: Se for melhor que o melhor global, aceita mesmo se for tabu
            is_aspirational = neighbor_fitness > best_fitness
            
            if neighbor_fitness > best_neighbor_fitness:
                if neighbor_tuple not in tabu_list or is_aspirational:
                    best_neighbor_x = neighbor_x
                    best_neighbor_fitness = neighbor_fitness

        if best_neighbor_x is None:
            # Todos os vizinhos explorados estão na lista tabu e não são melhores
            # Damos um passo aleatório para sair
            current_x = get_random_neighbor(current_x, n, p, B, k, aggressive=True)
            current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
        else:
            current_x = best_neighbor_x
            current_fitness = best_neighbor_fitness
            
            # Adiciona a solução atual à lista tabu
            tabu_list.append(tuple(current_x))
        
        # Atualiza o melhor global
        if current_fitness > best_fitness:
            best_x = current_x
            best_fitness = current_fitness
            
        history.append(best_fitness)

    return best_x, best_fitness, history

# --- 7. ALGORITMO 5: DISCRETE PARTICLE SWARM OPTIMIZATION (DPSO) (CORRIGIDO) ---

def discrete_pso(n, mu, sigma, p, B, k, lambda_val, alpha, beta,
                 n_particles=50, max_iter=200, w=0.5, c1=1.5, c2=1.5):
    
    particles = []
    for _ in range(n_particles):
        # FIX: Inicialização proporcional ao preço
        position = np.zeros(n)
        for j in range(n):
            if p[j] > 0:
                max_shares = k / p[j]
                position[j] = random.uniform(0, max_shares * 0.5)
                
        velocity = np.random.uniform(-1, 1, n)
        int_pos = np.round(position).astype(int)
        fitness = calculate_fitness(int_pos, mu, sigma, p, B, k, lambda_val, alpha, beta)
        
        particles.append({
            'position': position,
            'velocity': velocity,
            'pbest_position': int_pos,
            'pbest_fitness': fitness
        })

    # (O resto da função discrete_pso mantém-se igual ao que tinhas corrigido antes...)
    # ... (código do loop principal igual)
    
    # Certifica-te de copiar o resto da tua função discrete_pso original aqui
    # Apenas a inicialização acima ('for _ in range(n_particles)') é que mudou.
    
    # ... (Mantém o loop for particle in particles... etc)
    
    gbest_fitness = -np.inf
    gbest_position = np.zeros(n, dtype=int)
    
    for p_loop in particles: 
        if p_loop['pbest_fitness'] > gbest_fitness:
            gbest_fitness = p_loop['pbest_fitness']
            gbest_position = p_loop['pbest_position']
            
    history = [gbest_fitness]
    
    for i in range(max_iter):
        for particle in particles: 
            int_pos = np.round(particle['position']).astype(int)
            int_pos = np.maximum(0, int_pos)
            fitness = calculate_fitness(int_pos, mu, sigma, p, B, k, lambda_val, alpha, beta)
            
            if fitness > particle['pbest_fitness']:
                particle['pbest_fitness'] = fitness
                particle['pbest_position'] = int_pos
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = int_pos
                    
        for particle in particles: 
            r1 = random.random()
            r2 = random.random()
            current_pos_int = np.round(particle['position']).astype(int)
            cognitive_vel = c1 * r1 * (particle['pbest_position'] - current_pos_int)
            social_vel = c2 * r2 * (gbest_position - current_pos_int)
            new_velocity = w * particle['velocity'] + cognitive_vel + social_vel
            # Aumentar limite de velocidade para permitir saltos maiores
            new_velocity = np.clip(new_velocity, -10, 10) 
            particle['position'] = particle['position'] + new_velocity
            particle['velocity'] = new_velocity

        history.append(gbest_fitness)
        if (i + 1) % 50 == 0:
            print(f"  PSO Iteração {i+1}/{max_iter} - Melhor Fitness: {gbest_fitness:.4f}")

    return gbest_position, gbest_fitness, history


# --- 8. EXECUÇÃO E VISUALIZAÇÃO ---

def print_results(algorithm_name, x_star, p, mu, sigma, lambda_val, tickers):
    """ Imprime os resultados finais """
    # FIX: Passed 'p' to these functions
    final_return = calculate_return(x_star, mu, p)
    final_risk = calculate_risk(x_star, sigma, p)
    
    final_investment = np.dot(p, x_star)
    final_utility = final_return - lambda_val * final_risk
    
    print(f"\n--- Resultados para {algorithm_name} ---")
    print(f"Solução Final (x*): {x_star[x_star > 0]}") 
    print(f"Ativos Usados: {np.sum(x_star > 0)} de {len(tickers)}")
    print(f"Retorno Esperado: {final_return:.4f}")
    print(f"Risco (Variância): {final_risk:.4f}")
    print(f"Investimento Total: {final_investment:.2f} (Budget: {B})")
    print(f"Valor de Fitness (Utilidade): {final_utility:.4f}")
    
    # Verificação de Restrições
    violations_k = np.sum((p * x_star) > k)
    if violations_k > 0: print(f"  Restrição (k): {violations_k} ativos violaram o limite de ${k}.")
    else: print("  Restrição (k): OK.")
        
    if final_investment > B: print(f"  Restrição (B): Orçamento de ${B} violado.")
    else: print(f"  Restrição (B): OK.")

def plot_covariance_heatmap(sigma_df, tickers):
    """ (ENHANCED) Heatmap da matriz de covariância """
    if len(tickers) > 50:
        print("\nHeatmap: Demasiados ativos ( > 50) para um heatmap legível. A ser ignorado.")
        return
        
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Create heatmap with better colors
    sns.heatmap(sigma_df, annot=False, cmap='RdYlGn_r', 
                xticklabels=tickers, yticklabels=tickers,
                cbar_kws={'label': 'Covariância Anualizada'},
                linewidths=0.5, linecolor='white', ax=ax)
    
    plt.title('Heatmap da Covariância Anualizada dos Ativos\n(Vermelho = Alta correlação, Verde = Baixa correlação)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('heatmap_covariancia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'heatmap_covariancia.png' guardado.")

def plot_risk_return_scatter(results, mu, sigma, p):
    """ (ENHANCED) Gráfico de Risco vs Retorno com mais informação """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (name, x_star, fitness) in enumerate(results):
        # FIX: Passed 'p' here
        retorno = calculate_return(x_star, mu, p)
        risco = calculate_risk(x_star, sigma, p)
        
        # Plot main point
        ax.scatter(risco, retorno, s=300, 
                  color=colors[idx % len(colors)],
                  marker=markers[idx % len(markers)],
                  alpha=0.7, edgecolors='black', linewidths=2,
                  label=f'{name}', zorder=5)
        
        # Add annotation with fitness value
        ax.annotate(f'Fit: {fitness:.2f}', 
                   xy=(risco, retorno),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            fc=colors[idx % len(colors)], 
                            alpha=0.3))

    ax.set_title('Comparação de Portfólios: Risco vs. Retorno', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risco (Variância do Portfólio)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retorno Esperado ($)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('comparacao_risco_retorno.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'comparacao_risco_retorno.png' guardado.")

def plot_convergence(histories):
    """ (ENHANCED) Gráfico de convergência com múltiplas visualizações """
    n_algos = len(histories)
    n_rows = 2 + (n_algos + 1) // 2  # First row for combined, rest for individuals
    
    fig = plt.figure(figsize=(18, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot 1: All algorithms together (normalized)
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (name, history) in enumerate(histories):
        if len(history) > 1:
            x_axis = np.linspace(0, 100, len(history))
            ax1.plot(x_axis, history, label=name,
                    linewidth=2.5, color=colors[idx % len(colors)],
                    alpha=0.8)
    
    ax1.set_title('Convergência de Todos os Algoritmos (Eixo X Normalizado)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Progresso da Execução (%)', fontweight='bold')
    ax1.set_ylabel('Melhor Fitness Encontrado', fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-N: Individual algorithm plots
    for idx, (name, history) in enumerate(histories):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        if len(history) > 1:
            x_axis = np.arange(len(history))
            ax.plot(x_axis, history, linewidth=2,
                   color=colors[idx % len(colors)], alpha=0.8)
            ax.fill_between(x_axis, history, alpha=0.3,
                           color=colors[idx % len(colors)])
            
            # Add statistics
            final_fitness = history[-1]
            max_fitness = max(history)
            improvement = ((max_fitness - history[0]) / abs(history[0]) * 100
                          if history[0] != 0 else 0)
            
            stats_text = (f'Final: {final_fitness:.2f}\n'
                         f'Max: {max_fitness:.2f}\n'
                         f'Improvement: {improvement:.1f}%')
            
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        ax.set_title(f'{name}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Iteração / Geração', fontsize=9)
        ax.set_ylabel('Fitness', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Análise Detalhada da Convergência dos Algoritmos',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('convergencia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'convergencia.png' guardado.")

def plot_summary_table(results_data, mu, sigma, p, B, k):
    """ (NEW) Cria uma tabela visual com resumo estatístico """
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    headers = ['Algoritmo', 'Fitness', 'Retorno', 'Risco', 'Sharpe',
              'Investimento', '% Budget', 'N° Ativos', 'Violações']
    
    table_data = []
    for name, x, fitness in results_data:
        ret = calculate_return(x, mu, p) # Added p
        risk = calculate_risk(x, sigma, p) # Added p
        sharpe = ret / np.sqrt(risk) if risk > 0 else 0
        investment = np.dot(p, x)
        pct_budget = (investment / B) * 100
        n_assets = np.sum(x > 0)
        
        # Check violations
        budget_viol = max(0, investment - B)
        individual_invs = p * x
        limit_viol = np.sum(individual_invs > k)
        violations = f'{int(limit_viol)} ativos' if limit_viol > 0 else '✓ OK'
        if budget_viol > 0:
            violations = f'Budget! {violations}'
        
        table_data.append([
            name,
            f'{fitness:.2f}',
            f'{ret:.4f}',
            f'{risk:.4f}',
            f'{sharpe:.3f}',
            f'${investment:.0f}',
            f'{pct_budget:.1f}%',
            int(n_assets),
            violations
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.08, 0.1, 0.1, 0.08,
                              0.12, 0.09, 0.09, 0.19])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color rows by rank
    colors_rank = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    fitnesses = [fit for _, _, fit in results_data]
    ranks = np.argsort(np.argsort(fitnesses)[::-1])
    
    for i, rank in enumerate(ranks):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors_rank[rank % len(colors_rank)])
            cell.set_alpha(0.3)
            if j == 0:  # Algorithm name
                cell.set_text_props(weight='bold')
    
    plt.suptitle('Tabela Resumo Estatístico de Todos os Algoritmos\n'
                '(Ordenado por Fitness - Verde = Melhor)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('tabela_resumo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'tabela_resumo.png' guardado.")


def plot_radar_chart(results_data, mu, sigma, p, B, k, n):
    """ (NEW) Radar chart comparando algoritmos em múltiplas dimensões """
    from math import pi
    
    # Prepare metrics (normalized to 0-1 scale for comparison)
    metrics = ['Fitness', 'Retorno', 'Diversif.', 'Budget\nUsage', 
               'Baixo\nRisco', 'Sharpe']
    num_vars = len(metrics)
    
    # Calculate values for each algorithm
    algo_values = []
    algo_names = []
    
    for name, x, fitness in results_data:
        ret = calculate_return(x, mu, p) # Added p
        risk = calculate_risk(x, sigma, p) # Added p
        sharpe = ret / np.sqrt(risk) if risk > 0 else 0
        investment = np.dot(p, x)
        n_assets = np.sum(x > 0)
        
        algo_names.append(name)
        algo_values.append([fitness, ret, n_assets, investment, risk, sharpe])
    
    # Normalize each metric to 0-1 scale
    algo_values = np.array(algo_values)
    normalized = np.zeros_like(algo_values)
    
    for i in range(algo_values.shape[1]):
        col = algo_values[:, i]
        if i == 4:  # Risk - lower is better, so invert
            max_val, min_val = col.max(), col.min()
            if max_val != min_val:
                normalized[:, i] = 1 - (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5
        else:  # Higher is better
            max_val, min_val = col.max(), col.min()
            if max_val != min_val:
                normalized[:, i] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5
    
    # Create radar chart
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (name, values) in enumerate(zip(algo_names, normalized)):
        values = list(values) + [values[0]]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2.5, 
               label=name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Comparação Multi-Dimensional dos Algoritmos\n'
             '(Valores Normalizados: 1.0 = Melhor Performance)',
             fontsize=15, fontweight='bold', pad=30)
    
    plt.savefig('radar_comparacao.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'radar_comparacao.png' guardado.")


def plot_asset_popularity_heatmap(results_data, tickers, p):
    """ (NEW) Heatmap mostrando quais ativos são escolhidos por cada algoritmo """
    # Create matrix: rows = algorithms, columns = assets
    algo_names = [name for name, _, _ in results_data]
    n_algos = len(algo_names)
    n_assets = len(tickers)
    
    # Get top assets (those selected by at least one algorithm)
    asset_matrix = []
    for name, x, _ in results_data:
        asset_matrix.append(x)
    
    asset_matrix = np.array(asset_matrix)
    
    # Find assets that are used by at least one algorithm
    used_assets = np.any(asset_matrix > 0, axis=0)
    used_indices = np.where(used_assets)[0]
    
    if len(used_indices) > 60:
        # Too many assets, show only top invested ones
        total_investment = np.sum(asset_matrix * p, axis=0)
        top_indices = np.argsort(total_investment)[-60:][::-1]
    else:
        top_indices = used_indices
    
    # Create filtered matrix
    filtered_matrix = asset_matrix[:, top_indices]
    filtered_tickers = [tickers[i] for i in top_indices]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(18, len(filtered_tickers)*0.35), 10))
    
    im = ax.imshow(filtered_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(filtered_tickers)))
    ax.set_yticks(np.arange(n_algos))
    ax.set_xticklabels(filtered_tickers, rotation=90, ha='right', fontsize=9)
    ax.set_yticklabels(algo_names, fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Número de Ações', rotation=270, labelpad=20, 
                   fontsize=11, fontweight='bold')
    
    # Add values in cells (if not too many)
    if len(filtered_tickers) <= 40:
        for i in range(n_algos):
            for j in range(len(filtered_tickers)):
                value = filtered_matrix[i, j]
                if value > 0:
                    text = ax.text(j, i, int(value),
                                  ha="center", va="center", 
                                  color="black" if value < filtered_matrix.max()/2 
                                  else "white",
                                  fontsize=8, fontweight='bold')
    
    ax.set_title(f'Popularidade dos Ativos: Quais Ações Cada Algoritmo Escolheu\n'
                f'(Top {len(filtered_tickers)} ativos mais investidos)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Ativo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Algoritmo', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('heatmap_popularidade_ativos.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'heatmap_popularidade_ativos.png' guardado.")


def plot_convergence_speed(histories):
    """ (NEW) Análise da velocidade de convergência """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    algo_names = []
    iterations_to_90 = []
    iterations_to_95 = []
    final_fitness = []
    
    for idx, (name, history) in enumerate(histories):
        algo_names.append(name)
        
        if len(history) < 2:
            iterations_to_90.append(0)
            iterations_to_95.append(0)
            final_fitness.append(history[0] if len(history) > 0 else 0)
            continue
        
        final_fit = history[-1]
        final_fitness.append(final_fit)
        target_90 = final_fit * 0.90
        target_95 = final_fit * 0.95
        
        # Find when it reaches 90% and 95%
        iter_90 = len(history)
        iter_95 = len(history)
        
        for i, fit in enumerate(history):
            if fit >= target_90 and iter_90 == len(history):
                iter_90 = i
            if fit >= target_95 and iter_95 == len(history):
                iter_95 = i
        
        iterations_to_90.append(iter_90)
        iterations_to_95.append(iter_95)
    
    # Plot 1: Iterations to reach targets
    x = np.arange(len(algo_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, iterations_to_90, width, 
                   label='90% do Fitness Final', color='#3498db', 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, iterations_to_95, width,
                   label='95% do Fitness Final', color='#2ecc71',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Iterações / Gerações', fontweight='bold', fontsize=11)
    ax1.set_title('Velocidade de Convergência\n(Menos iterações = Mais rápido)',
                 fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_names, rotation=20, ha='right')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Convergence efficiency (final fitness / iterations)
    efficiency = []
    total_iters = []
    
    for idx, (name, history) in enumerate(histories):
        total_iters.append(len(history))
        if len(history) > 0:
            eff = final_fitness[idx] / len(history)
            efficiency.append(eff)
        else:
            efficiency.append(0)
    
    bars = ax2.barh(algo_names, efficiency, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Eficiência (Fitness / Iteração)', fontweight='bold', fontsize=11)
    ax2.set_title('Eficiência dos Algoritmos\n(Maior = Melhor aproveitamento)',
                 fontweight='bold', fontsize=13)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val, iters in zip(bars, efficiency, total_iters):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.3f} ({iters} iter)',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('velocidade_convergencia.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'velocidade_convergencia.png' guardado.")


def plot_top_assets_analysis(results_data, tickers, p):
    """ (NEW) Análise dos ativos mais populares """
    # Aggregate investment across all algorithms
    total_shares = np.zeros(len(tickers))
    total_value = np.zeros(len(tickers))
    selection_count = np.zeros(len(tickers))
    
    for name, x, _ in results_data:
        total_shares += x
        total_value += x * p
        selection_count += (x > 0).astype(int)
    
    # Get top 20 assets by total value
    top_indices = np.argsort(total_value)[-20:][::-1]
    top_tickers = [tickers[i] for i in top_indices]
    top_values = total_value[top_indices]
    top_counts = selection_count[top_indices]
    top_shares = total_shares[top_indices]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Investment value
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_tickers)))
    bars1 = ax1.bar(top_tickers, top_values, color=colors_grad,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax1.set_ylabel('Valor Total Investido ($)', fontweight='bold', fontsize=12)
    ax1.set_title('Top 20 Ativos Mais Investidos (Soma de Todos os Algoritmos)',
                 fontweight='bold', fontsize=14, pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars1, top_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Selection frequency
    colors_freq = ['#2ecc71' if c == len(results_data) else '#3498db' if c >= len(results_data)/2 else '#e67e22' 
                   for c in top_counts]
    bars2 = ax2.bar(top_tickers, top_counts, color=colors_freq,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax2.set_ylabel('Número de Algoritmos que Selecionaram', 
                  fontweight='bold', fontsize=12)
    ax2.set_xlabel('Ativo', fontweight='bold', fontsize=12)
    ax2.set_title('Frequência de Seleção dos Top 20 Ativos\n'
                 '(Verde = Todos | Azul = Maioria | Laranja = Alguns)',
                 fontweight='bold', fontsize=14, pad=15)
    ax2.set_ylim(0, len(results_data) + 0.5)
    ax2.axhline(y=len(results_data), color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label='Todos os algoritmos')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, val, shares in zip(bars2, top_counts, top_shares):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}/{len(results_data)}\n({int(shares)} ações)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('top_ativos_analise.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'top_ativos_analise.png' guardado.")


def plot_portfolio_composition_comparison(results_data, tickers, p):
    """ (NEW) Comparação lado-a-lado da composição dos portfólios """
    # Get top 15 most invested assets across all algorithms
    total_value = np.zeros(len(tickers))
    
    for name, x, _ in results_data:
        total_value += x * p
    
    top_indices = np.argsort(total_value)[-15:][::-1]
    top_tickers = [tickers[i] for i in top_indices]
    
    # Create matrix for stacked bar chart
    n_algos = len(results_data)
    n_assets = len(top_tickers)
    
    data_matrix = np.zeros((n_assets, n_algos))
    algo_names = []
    
    for algo_idx, (name, x, _) in enumerate(results_data):
        algo_names.append(name)
        for asset_idx, ticker_idx in enumerate(top_indices):
            data_matrix[asset_idx, algo_idx] = x[ticker_idx] * p[ticker_idx]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x_pos = np.arange(n_algos)
    colors = plt.cm.tab20(np.linspace(0, 1, n_assets))
    
    bottom = np.zeros(n_algos)
    
    for asset_idx in range(n_assets):
        bars = ax.bar(x_pos, data_matrix[asset_idx], 0.8, 
                     label=top_tickers[asset_idx],
                     bottom=bottom, color=colors[asset_idx],
                     edgecolor='white', linewidth=0.5)
        bottom += data_matrix[asset_idx]
    
    ax.set_ylabel('Valor Investido ($)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Algoritmo', fontweight='bold', fontsize=12)
    ax.set_title('Comparação da Composição dos Portfólios\n'
                '(Top 15 Ativos Mais Investidos)',
                fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), 
             title='Ativos', fontsize=9, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total value on top of each bar
    for i, total in enumerate(bottom):
        ax.text(i, total, f'${total:.0f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('composicao_portfolios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'composicao_portfolios.png' guardado.")


def plot_efficient_frontier(results_data, mu, sigma, p):
    """ (NEW) Fronteira eficiente aproximada """
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    risks = []
    returns = []
    names_list = []
    
    for idx, (name, x, fitness) in enumerate(results_data):
        # FIX: Passed 'p' here
        ret = calculate_return(x, mu, p)
        risk = calculate_risk(x, sigma, p)
        
        risks.append(risk)
        returns.append(ret)
        names_list.append(name)
        
        ax.scatter(risk, ret, s=400, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], alpha=0.7, edgecolors='black', linewidths=2.5, label=name, zorder=5)
        
        ax.annotate(name, xy=(risk, ret), xytext=(15, 15), textcoords='offset points', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', fc=colors[idx % len(colors)], alpha=0.2))
    
    # (Rest of the plotting logic remains similar, just ensuring 'p' was passed above)
    
    ax.set_xlabel('Risco ($^2)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Retorno Esperado ($)', fontweight='bold', fontsize=13)
    ax.set_title('Fronteira Eficiente: Soluções Pareto-Ótimas', fontweight='bold', fontsize=16, pad=20)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('fronteira_eficiente.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'fronteira_eficiente.png' guardado.")


def plot_algorithm_comparison_dashboard(results_data, mu, sigma, p, B, k):
    """ (NEW) Cria uma tabela visual com resumo estatístico """
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    headers = ['Algoritmo', 'Fitness', 'Retorno', 'Risco', 'Sharpe',
              'Investimento', '% Budget', 'N° Ativos', 'Violações']
    
    table_data = []
    for name, x, fitness in results_data:
        ret = calculate_return(x, mu)
        risk = calculate_risk(x, sigma)
        sharpe = ret / np.sqrt(risk) if risk > 0 else 0
        investment = np.dot(p, x)
        pct_budget = (investment / B) * 100
        n_assets = np.sum(x > 0)
        
        # Check violations
        budget_viol = max(0, investment - B)
        individual_invs = p * x
        limit_viol = np.sum(individual_invs > k)
        violations = f'{int(limit_viol)} ativos' if limit_viol > 0 else '✓ OK'
        if budget_viol > 0:
            violations = f'Budget! {violations}'
        
        table_data.append([
            name,
            f'{fitness:.2f}',
            f'{ret:.4f}',
            f'{risk:.4f}',
            f'{sharpe:.3f}',
            f'${investment:.0f}',
            f'{pct_budget:.1f}%',
            int(n_assets),
            violations
        ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.08, 0.1, 0.1, 0.08,
                              0.12, 0.09, 0.09, 0.19])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color rows by rank
    colors_rank = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    fitnesses = [fit for _, _, fit in results_data]
    ranks = np.argsort(np.argsort(fitnesses)[::-1])
    
    for i, rank in enumerate(ranks):
        for j in range(len(headers)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors_rank[rank % len(colors_rank)])
            cell.set_alpha(0.3)
            if j == 0:  # Algorithm name
                cell.set_text_props(weight='bold')
    
    plt.suptitle('Tabela Resumo Estatístico de Todos os Algoritmos\n'
                '(Ordenado por Fitness - Verde = Melhor)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('tabela_resumo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'tabela_resumo.png' guardado.")


def plot_algorithm_comparison_dashboard(results_data, mu, sigma, p, B, k):
    """ (NEW) Cria um dashboard completo comparando todos os algoritmos """
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.4)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    algo_names = [name for name, _, _ in results_data]
    
    # 1. Fitness Comparison Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    fitnesses = [fit for _, _, fit in results_data]
    bars = ax1.barh(algo_names, fitnesses, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Fitness (Utilidade - Penalidade)', fontweight='bold')
    ax1.set_title('Comparação de Fitness', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, fitnesses):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.1f}', va='center', fontweight='bold', fontsize=8)
    
    # 2. Return Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    returns = [calculate_return(x, mu, p) for _, x, _ in results_data] # Added p
    bars = ax2.barh(algo_names, returns, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Retorno Esperado Anualizado', fontweight='bold')
    ax2.set_title('Comparação de Retorno', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, returns):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.3f}', va='center', fontweight='bold', fontsize=8)
    
    # 3. Risk Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    risks = [calculate_risk(x, sigma, p) for _, x, _ in results_data] # Added p
    bars = ax3.barh(algo_names, risks, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Risco (Variância)', fontweight='bold')
    ax3.set_title('Comparação de Risco', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, risks):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.2f}', va='center', fontweight='bold', fontsize=8)
    
    # 4. Investment Amount Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    investments = [np.dot(p, x) for _, x, _ in results_data]
    bars = ax4.barh(algo_names, investments, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax4.axvline(x=B, color='red', linestyle='--', linewidth=2,
               label=f'Orçamento = ${B:.0f}', alpha=0.7)
    ax4.set_xlabel('Investimento Total ($)', fontweight='bold')
    ax4.set_title('Uso do Orçamento', fontweight='bold', fontsize=12)
    ax4.legend(loc='best')
    ax4.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, investments):
        width = bar.get_width()
        pct = (val / B) * 100
        ax4.text(width, bar.get_y() + bar.get_height()/2,
                f' ${val:.0f} ({pct:.0f}%)',
                va='center', fontweight='bold', fontsize=7)
    
    # 5. Number of Assets Used
    ax5 = fig.add_subplot(gs[1, 1])
    n_assets = [np.sum(x > 0) for _, x, _ in results_data]
    bars = ax5.barh(algo_names, n_assets, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Número de Ativos Diferentes', fontweight='bold')
    ax5.set_title('Diversificação do Portfólio', fontweight='bold', fontsize=12)
    ax5.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, n_assets):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(val)}', va='center', fontweight='bold', fontsize=8)
    
    # 6. Sharpe Ratio Comparison (Risk-adjusted return)
    ax6 = fig.add_subplot(gs[1, 2])
    sharpe_ratios = []
    for _, x, _ in results_data:
        ret = calculate_return(x, mu, p) # Added p
        risk = calculate_risk(x, sigma, p) # Added p
        sharpe = ret / np.sqrt(risk) if risk > 0 else 0
        sharpe_ratios.append(sharpe)
    
    bars = ax6.barh(algo_names, sharpe_ratios, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Sharpe Ratio (aprox.)', fontweight='bold')
    ax6.set_title('Retorno Ajustado ao Risco', fontweight='bold', fontsize=12)
    ax6.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, sharpe_ratios):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.2f}', va='center', fontweight='bold', fontsize=8)
    
    # 7. Constraint Violations
    ax7 = fig.add_subplot(gs[2, :])
    
    budget_violations = []
    limit_violations = []
    
    for _, x, _ in results_data:
        total_inv = np.dot(p, x)
        budget_viol = max(0, total_inv - B)
        budget_violations.append(budget_viol)
        
        individual_invs = p * x
        limit_viol = np.sum(np.maximum(0, individual_invs - k))
        limit_violations.append(limit_viol)
    
    x_pos = np.arange(len(algo_names))
    width = 0.35
    
    bars1 = ax7.bar(x_pos - width/2, budget_violations, width,
                   label='Violação de Orçamento', color='#e74c3c',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax7.bar(x_pos + width/2, limit_violations, width,
                   label='Violação de Limite por Ativo', color='#f39c12',
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax7.set_ylabel('Penalidade ($)', fontweight='bold')
    ax7.set_title('Violações de Restrições (Menor é Melhor)',
                 fontweight='bold', fontsize=12)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(algo_names, rotation=15, ha='right')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.suptitle('Dashboard de Comparação Completa dos Algoritmos',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('dashboard_comparacao.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico 'dashboard_comparacao.png' guardado.")


def plot_allocation_charts(x_star, p, tickers, algorithm_name, B, k):
    """ (ENHANCED) Gera gráficos de alocação mais informativos """
    
    # Filtra para mostrar apenas valores > 0.01 (ignora pó)
    values = p * x_star
    mask = values > 0.01
    values_gt_zero = values[mask]
    tickers_gt_zero = [tickers[i] for i, m in enumerate(mask) if m]
    x_gt_zero = x_star[mask]
    total_value = sum(values_gt_zero)
    
    if len(values_gt_zero) == 0:
        print(f"Gráficos de Alocação ({algorithm_name}): "
              f"Nenhuma ação selecionada.")
        return

    # Sort by value for better visualization
    sorted_indices = np.argsort(values_gt_zero)[::-1]
    values_sorted = values_gt_zero[sorted_indices]
    tickers_sorted = [tickers_gt_zero[i] for i in sorted_indices]
    x_sorted = x_gt_zero[sorted_indices]
    
    # --- Gráfico 1: N.º de Ações (Enhanced) ---
    fig, ax = plt.subplots(figsize=(max(14, len(tickers_sorted)*0.6), 7))
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(tickers_sorted)))
    bars = ax.bar(tickers_sorted, x_sorted, color=colors_gradient,
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, x_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(val)}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_title(f'Alocação (Nº de Ações) - {algorithm_name}\n'
                f'Total de {len(tickers_sorted)} ativos diferentes',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Número de Ações (x_i)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Ativo', fontweight='bold', fontsize=11)
    plt.xticks(rotation=90, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    filename = f'alocacao_acoes_{algorithm_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico '{filename}' guardado.")

    # --- Gráfico 2: Valor Monetário (Enhanced) ---
    fig, ax = plt.subplots(figsize=(max(14, len(tickers_sorted)*0.6), 8))
    
    # Color code bars: green if under limit, red if over
    bar_colors = ['#2ecc71' if v <= k else '#e74c3c' for v in values_sorted]
    bars = ax.bar(tickers_sorted, values_sorted, color=bar_colors,
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values_sorted):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${val:.0f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Reference lines
    ax.axhline(y=k, color='darkred', linestyle='--', linewidth=2,
              label=f'Limite por Ativo (k = ${k:.0f})', alpha=0.7)
    
    budget_usage = (total_value / B) * 100
    title = (f'Alocação (Valor Investido) - {algorithm_name}\n'
            f'Total: ${total_value:.2f} / ${B:.2f} '
            f'({budget_usage:.1f}% do orçamento)')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Valor Investido ($)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Ativo', fontweight='bold', fontsize=11)
    plt.xticks(rotation=90, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    filename = f'alocacao_valor_{algorithm_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico '{filename}' guardado.")

    # --- Gráfico 3: Pie Chart (Enhanced) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Agrupa fatias pequenas (<2%) em 'Outros'
    pct = values_sorted / total_value
    small_slices = pct < 0.02
    
    if np.any(small_slices):
        other_val = np.sum(values_sorted[small_slices])
        main_vals = list(values_sorted[~small_slices]) + [other_val]
        main_labels = (list(np.array(tickers_sorted)[~small_slices]) +
                      [f'Outros ({np.sum(small_slices)} ativos)'])
        main_pct = list(pct[~small_slices]) + [other_val / total_value]
    else:
        main_vals = values_sorted
        main_labels = tickers_sorted
        main_pct = pct
    
    # Pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(main_vals)))
    wedges, texts, autotexts = ax1.pie(main_vals, labels=main_labels,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors_pie,
                                       textprops={'fontsize': 9})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax1.set_title(f'Composição do Portfólio (%) - {algorithm_name}',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Donut chart version
    wedges2, texts2, autotexts2 = ax2.pie(main_vals, labels=main_labels,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          colors=colors_pie,
                                          wedgeprops=dict(width=0.5),
                                          textprops={'fontsize': 9})
    
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Add center text
    ax2.text(0, 0, f'Total\n${total_value:.0f}',
            ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    ax2.set_title(f'Composição do Portfólio (Donut) - {algorithm_name}',
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    filename = f'alocacao_pie_{algorithm_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico '{filename}' guardado.")


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    
    archive_path = 'archive/'
    selected_assets = []
    
    try:
        all_files_in_dir = os.listdir(archive_path)
        selected_assets = [f for f in all_files_in_dir if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Erro: A pasta '{archive_path}' não foi encontrada.")
    except Exception as e:
        print(f"Ocorreu um erro ao ler a pasta {archive_path}: {e}")

    
    # 2. Obter dados do problema
    # p, mu, sigma (já é np.array), n, tickers
    p, mu, sigma, n, tickers = get_problem_data(selected_assets)
    
    
    # 3. Executar os algoritmos (só se tivermos dados)
    if n > 0:
        print(f"\n--- A processar {n} ativos encontrados na pasta 'archive' ---")
        
        # --- Parâmetros dos Algoritmos (Ajustáveis) ---
        HC_MAX_ITER = 3000
        SA_MAX_ITER = 40000
        GA_GENERATIONS = 150
        GA_POP_SIZE = 80
        TS_MAX_ITER = 3000
        TS_TABU_TENURE = int(n * 0.5) # Lista tabu de 50 (para n=100)
        PSO_MAX_ITER = 150
        PSO_N_PARTICLES = 80
        
        # Simulated Annealing Strategy:
        # aggressive_neighbors=False: Mais conservador, explora melhor o espaço
        #                             Pode ter melhor posição na fronteira Pareto
        #                             Usa ~93-97% do budget
        # aggressive_neighbors=True:  Mais agressivo, usa mais budget
        #                             Pode ter maior fitness absoluto
        #                             Usa ~99% do budget
        SA_AGGRESSIVE = True  # ← MUDE AQUI para experimentar
        # ------------------------------------
        
        print("\n(1/5) A executar Hill Climbing...")
        hc_x, hc_fit, hc_hist = hill_climbing(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=HC_MAX_ITER
        )
        
        print("\n(2/5) A executar Simulated Annealing...")
        sa_x, sa_fit, sa_hist = simulated_annealing(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=SA_MAX_ITER,
            aggressive_neighbors=SA_AGGRESSIVE
        )
        
        print("\n(3/5) A executar Genetic Algorithm...")
        ga_x, ga_fit, ga_hist = genetic_algorithm(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta,
            pop_size=GA_POP_SIZE, generations=GA_GENERATIONS
        )
        
        print("\n(4/5) A executar Tabu Search...")
        ts_x, ts_fit, ts_hist = tabu_search(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta,
            max_iter=TS_MAX_ITER, tabu_tenure=TS_TABU_TENURE
        )

        print("\n(5/5) A executar Particle Swarm Optimization (PSO)...")
        pso_x, pso_fit, pso_hist = discrete_pso(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta,
            n_particles=PSO_N_PARTICLES, max_iter=PSO_MAX_ITER
        )

        # 5. Apresentar Resultados
        print("\n\n" + "="*30 + " SUMÁRIO DOS RESULTADOS " + "="*30)
        
        results_data = [
            ("Hill Climbing", hc_x, hc_fit),
            ("Simulated Annealing", sa_x, sa_fit),
            ("Genetic Algorithm", ga_x, ga_fit),
            ("Tabu Search", ts_x, ts_fit),
            ("Particle Swarm (PSO)", pso_x, pso_fit)
        ]

        # Imprime o sumário para cada
        for name, x, fit in results_data:
            print_results(name, x, p, mu, sigma, lambda_val, tickers)
        
        # Encontra o algoritmo com o maior fitness (APENAS PARA O SUMÁRIO)
        best_algo_name, best_x, best_fit = max(results_data, key=lambda item: item[2])
        
        print("\n" + "="*50)
        print(f"🏆 MELHOR SOLUÇÃO ENCONTRADA FOI DE: {best_algo_name}")
        print(f"   Com um Fitness (Utilidade - Penalidade) de: {best_fit:.4f}")
        print("="*50)

        # 6. Gerar Gráficos
        
        # --- Gráficos Comparativos (Todos os algoritmos juntos) ---
        print("\n" + "="*60)
        print("A GERAR VISUALIZAÇÕES AVANÇADAS")
        print("="*60)
        
        print("\n[1/10] Tabela Resumo Estatístico...")
        plot_summary_table(results_data, mu, sigma, p, B, k)
        
        print("\n[2/10] Radar Chart Multi-Dimensional...")
        plot_radar_chart(results_data, mu, sigma, p, B, k, n)
        
        print("\n[3/10] Dashboard de Comparação Completa...")
        plot_algorithm_comparison_dashboard(results_data, mu, sigma, p, B, k)
        
        print("\n[4/10] Gráfico de Convergência Detalhado...")
        plot_convergence([
            ("Hill Climbing", hc_hist),
            ("Simulated Annealing", sa_hist),
            ("Genetic Algorithm", ga_hist),
            ("Tabu Search", ts_hist),
            ("Particle Swarm (PSO)", pso_hist)
        ])
        
        print("\n[5/10] Análise de Velocidade de Convergência...")
        plot_convergence_speed([
            ("Hill Climbing", hc_hist),
            ("Simulated Annealing", sa_hist),
            ("Genetic Algorithm", ga_hist),
            ("Tabu Search", ts_hist),
            ("Particle Swarm (PSO)", pso_hist)
        ])
        
        print("\n[6/10] Fronteira Eficiente...")
        plot_efficient_frontier(results_data, mu, sigma, p)
        
        print("\n[7/10] Análise Risco vs Retorno...")
        plot_risk_return_scatter(results_data, mu, sigma, p)
        
        print("\n[8/10] Heatmap de Popularidade dos Ativos...")
        plot_asset_popularity_heatmap(results_data, tickers, p)
        
        print("\n[9/10] Análise dos Top Ativos...")
        plot_top_assets_analysis(results_data, tickers, p)
        
        print("\n[10/10] Comparação de Composição dos Portfólios...")
        plot_portfolio_composition_comparison(results_data, tickers, p)
        
        # --- Gráficos de Alocação (Um conjunto para CADA algoritmo) ---
        print("\n" + "="*60)
        print("GRÁFICOS DE ALOCAÇÃO INDIVIDUAL")
        print("="*60)
        
        for idx, (name, x, fit) in enumerate(results_data, 1):
            print(f"\n[{idx}/{len(results_data)}] Gerando: {name}")
            plot_allocation_charts(x, p, tickers, name, B, k)
        
        total_graphs = 10 + len(results_data) * 3 + 1  # +1 for heatmap
        print("\n" + "="*60)
        print("✓ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"✓ Total de gráficos gerados: {total_graphs}")
        print("✓ Resolução: 300 DPI (alta qualidade)")
        print("="*60)
        print("\nNOVOS Gráficos Avançados:")
        print("  • radar_comparacao.png - Comparação multi-dimensional")
        print("  • velocidade_convergencia.png - Análise de velocidade")
        print("  • fronteira_eficiente.png - Fronteira eficiente")
        print("  • heatmap_popularidade_ativos.png - Popularidade de ativos")
        print("  • top_ativos_analise.png - Top 20 ativos")
        print("  • composicao_portfolios.png - Composição lado-a-lado")
        print("\nGráficos Principais:")
        print("  • tabela_resumo.png - Tabela estatística comparativa")
        print("  • dashboard_comparacao.png - Dashboard completo")
        print("  • convergencia.png - Análise de convergência detalhada")
        print("  • comparacao_risco_retorno.png - Scatter plot risco/retorno")
        print("  • heatmap_covariancia.png - Matriz de covariância")
        print("  • alocacao_*.png - 3 gráficos por algoritmo (15 total)")
        print("="*60)
        
    else:
        print("\nNenhum dado foi carregado. A execução foi interrompida.")
        if not selected_assets:
            print("Verifique se a pasta 'archive' existe e contém "
                  "ficheiros .csv.")
