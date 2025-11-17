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
            tickers.append(ticker)
            
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            
            last_price = df['Close'].iloc[-1]
            prices.append(last_price)
            
            daily_returns = df['Close'].pct_change(fill_method=None).dropna()
            all_returns.append(daily_returns)
            
        except FileNotFoundError:
            print(f"Erro: Ficheiro não encontrado em {filepath}")
            return None, None, None, None, None
        except Exception as e:
            print(f"Erro ao processar {filepath}: {e}")
            return None, None, None, None, None

    returns_df = pd.concat(all_returns, axis=1, keys=tickers)
    returns_df = returns_df.dropna()
    
    mu = returns_df.mean().values * 252
    sigma = returns_df.cov() * 252 # Manter como DataFrame para o heatmap
    
    p = np.array(prices)
    
    print("--- Dados de Problema Carregados ---")
    print(f"N (Número de ativos): {n}")
    
    # Gerar o heatmap da covariância (NOVA VISUALIZAÇÃO)
    plot_covariance_heatmap(sigma, tickers)
    
    # Converter sigma para numpy array para os cálculos
    sigma_np = sigma.values
    
    print(f"Sigma (Covariância Anualizada):\n{sigma_np[:2, :2]} ... (apenas 2x2)")
    print("------------------------------------")
    
    return p, mu, sigma_np, n, tickers

# --- 2. DEFINIÇÃO DO PROBLEMA (IA.pdf) ---

B = 5000.0   # Orçamento total
k = 1000.0   # Investimento máximo por ativo
lambda_val = 0.1  # Aversão ao risco
alpha = 1000.0 # Penalidade de orçamento
beta = 1000.0  # Penalidade de limite por ativo

def calculate_return(x, mu):
    return np.dot(mu, x)

def calculate_risk(x, sigma):
    return np.dot(x.T, np.dot(sigma, x))

def calculate_penalty(x, p, B, k, alpha, beta):
    total_investment = np.dot(p, x)
    penalty_B = alpha * max(0, total_investment - B)
    
    individual_investments = p * x
    violations_k = [max(0, inv - k) for inv in individual_investments]
    penalty_k = beta * sum(violations_k)
    
    return penalty_B + penalty_k

def calculate_fitness(x, mu, sigma, p, B, k, lambda_val, alpha, beta):
    # Assegurar que x é não-negativo
    x = np.maximum(0, x)
    
    retorno = calculate_return(x, mu)
    risco = calculate_risk(x, sigma)
    utility = retorno - lambda_val * risco
    penalty = calculate_penalty(x, p, B, k, alpha, beta)
    return utility - penalty

# --- 3. ALGORITMO 1: HILL CLIMBING ---

def hill_climbing(n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=1000):
    current_x = np.zeros(n, dtype=int)
    current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
    history = [current_fitness]
    
    for _ in range(max_iter):
        best_neighbor_x = current_x
        best_neighbor_fitness = current_fitness
        
        for j in range(n):
            # Tenta +1
            x_plus = np.copy(current_x); x_plus[j] += 1
            fit_plus = calculate_fitness(x_plus, mu, sigma, p, B, k, lambda_val, alpha, beta)
            if fit_plus > best_neighbor_fitness:
                best_neighbor_x = x_plus; best_neighbor_fitness = fit_plus
                
            # Tenta -1
            if current_x[j] > 0:
                x_minus = np.copy(current_x); x_minus[j] -= 1
                fit_minus = calculate_fitness(x_minus, mu, sigma, p, B, k, lambda_val, alpha, beta)
                if fit_minus > best_neighbor_fitness:
                    best_neighbor_x = x_minus; best_neighbor_fitness = fit_minus

        if best_neighbor_fitness <= current_fitness:
            break
            
        current_x = best_neighbor_x
        current_fitness = best_neighbor_fitness
        history.append(current_fitness)
        
    return current_x, current_fitness, history

# --- 4. ALGORITMO 2: SIMULATED ANNEALING ---

def get_random_neighbor(x, n):
    x_neighbor = np.copy(x)
    i = random.randint(0, n - 1)
    if random.random() < 0.5: x_neighbor[i] += 1
    else: x_neighbor[i] -= 1
    x_neighbor[i] = max(0, x_neighbor[i])
    return x_neighbor

def simulated_annealing(n, mu, sigma, p, B, k, lambda_val, alpha, beta, 
                        max_iter=10000, T_initial=1000.0, T_final=0.1, cooling_rate=0.99):
    current_x = np.zeros(n, dtype=int)
    current_fitness = calculate_fitness(current_x, mu, sigma, p, B, k, lambda_val, alpha, beta)
    best_x = current_x
    best_fitness = current_fitness
    T = T_initial
    history = []
    
    for i in range(max_iter):
        if T <= T_final: break
        new_x = get_random_neighbor(current_x, n)
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

def ga_create_individual(n):
    return np.random.randint(0, 5, n, dtype=int)

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

def ga_mutation(individual, n, mutation_rate=0.05):
    if random.random() < mutation_rate:
        index = random.randint(0, n - 1)
        if random.random() < 0.5: individual[index] += 1
        else: individual[index] -= 1
        individual[index] = max(0, individual[index])
    return individual

def genetic_algorithm(n, mu, sigma, p, B, k, lambda_val, alpha, beta,
                      pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.1):
    population = [ga_create_individual(n) for _ in range(pop_size)]
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
            child = ga_mutation(child, n, mutation_rate)
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

def get_neighbors_ts(x, n, n_neighbors):
    """ Gera um conjunto de vizinhos para o Tabu Search """
    neighbors = []
    for _ in range(n_neighbors):
        neighbors.append(get_random_neighbor(x, n)) # Reutiliza a função do SA
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
        neighbors = get_neighbors_ts(current_x, n, n_neighbors)
        
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
            current_x = get_random_neighbor(current_x, n)
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
    
    # Inicializa as partículas
    particles = []
    for _ in range(n_particles):
        position = np.random.uniform(0, 5, n) # Posição inicial contínua
        velocity = np.random.uniform(-1, 1, n)
        
        # Avalia a posição inicial (discreta)
        int_pos = np.round(position).astype(int)
        # 'p' aqui refere-se ao array de preços (correto, pois não há conflito)
        fitness = calculate_fitness(int_pos, mu, sigma, p, B, k, lambda_val, alpha, beta)
        
        particles.append({
            'position': position,
            'velocity': velocity,
            'pbest_position': int_pos,
            'pbest_fitness': fitness
        })

    # Inicializa o melhor global
    gbest_fitness = -np.inf
    gbest_position = np.zeros(n, dtype=int)
    
    # Renomeado 'p' para 'p_loop' para evitar qualquer confusão
    for p_loop in particles: 
        if p_loop['pbest_fitness'] > gbest_fitness:
            gbest_fitness = p_loop['pbest_fitness']
            gbest_position = p_loop['pbest_position']
            
    history = [gbest_fitness]
    
    for i in range(max_iter):
        # === INÍCIO DA CORREÇÃO ===
        # O loop principal agora usa 'particle' para evitar conflito com 'p' (preços)
        for particle in particles: 
            # 1. Avalia o fitness da posição atual (discreta)
            int_pos = np.round(particle['position']).astype(int) # MUDADO de p['position']
            int_pos = np.maximum(0, int_pos) # Garante não-negativo
            
            # 'p' aqui AGORA refere-se corretamente ao array de preços 
            # (passado como argumento da função) e não à variável 'particle' do loop.
            fitness = calculate_fitness(int_pos, mu, sigma, p, B, k, lambda_val, alpha, beta)
            
            # 2. Atualiza pbest
            if fitness > particle['pbest_fitness']: # MUDADO de p['pbest_fitness']
                particle['pbest_fitness'] = fitness
                particle['pbest_position'] = int_pos
                
                # 3. Atualiza gbest
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = int_pos
                    
        # 4. Atualiza velocidade e posição de todas as partículas
        # Este loop também foi corrigido para 'particle'
        for particle in particles: 
            r1 = random.random()
            r2 = random.random()
            
            # Posição atual (discreta) para o cálculo da velocidade
            current_pos_int = np.round(particle['position']).astype(int) # MUDADO
            
            cognitive_vel = c1 * r1 * (particle['pbest_position'] - current_pos_int) # MUDADO
            social_vel = c2 * r2 * (gbest_position - current_pos_int)
            
            new_velocity = w * particle['velocity'] + cognitive_vel + social_vel # MUDADO
            
            # Limita a velocidade para evitar "explosão"
            new_velocity = np.clip(new_velocity, -3, 3) 
            
            particle['position'] = particle['position'] + new_velocity # MUDADO
            particle['velocity'] = new_velocity # MUDADO
        # === FIM DA CORREÇÃO ===

        history.append(gbest_fitness)
        if (i + 1) % 50 == 0:
            print(f"  PSO Iteração {i+1}/{max_iter} - Melhor Fitness: {gbest_fitness:.4f}")

    return gbest_position, gbest_fitness, history


# --- 8. EXECUÇÃO E VISUALIZAÇÃO ---

def print_results(algorithm_name, x_star, p, mu, sigma, lambda_val, tickers):
    """ Imprime os resultados finais """
    final_return = calculate_return(x_star, mu)
    final_risk = calculate_risk(x_star, sigma)
    final_investment = np.dot(p, x_star)
    final_utility = final_return - lambda_val * final_risk
    
    print(f"\n--- Resultados para {algorithm_name} ---")
    print(f"Solução Final (x*): {x_star[x_star > 0]}") # Mostra apenas > 0
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

def plot_risk_return_scatter(results, mu, sigma):
    """ (ENHANCED) Gráfico de Risco vs Retorno com mais informação """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (name, x_star, fitness) in enumerate(results):
        retorno = calculate_return(x_star, mu)
        risco = calculate_risk(x_star, sigma)
        
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

    ax.set_title('Comparação de Portfólios: Risco vs. Retorno\n' + 
                 'Objetivo: Maximizar Retorno e Minimizar Risco',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Risco (Variância do Portfólio)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retorno Esperado Anualizado', fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Add Sharpe ratio reference lines
    if len(results) > 0:
        max_ret = max([calculate_return(x, mu) for _, x, _ in results])
        max_risk = max([calculate_risk(x, sigma) for _, x, _ in results])
        
        # Draw diagonal lines for different Sharpe ratios
        x_line = np.linspace(0, max_risk * 1.1, 100)
        for sharpe in [0.5, 1.0, 1.5]:
            y_line = sharpe * np.sqrt(x_line)
            ax.plot(x_line, y_line, ':', alpha=0.3, linewidth=1,
                   label=f'Sharpe ≈ {sharpe}' if sharpe == 1.0 else '')
    
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
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
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
                f' {val:.2f}', va='center', fontweight='bold', fontsize=9)
    
    # 2. Return Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    returns = [calculate_return(x, mu) for _, x, _ in results_data]
    bars = ax2.barh(algo_names, returns, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Retorno Esperado Anualizado', fontweight='bold')
    ax2.set_title('Comparação de Retorno', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, returns):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}', va='center', fontweight='bold', fontsize=9)
    
    # 3. Risk Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    risks = [calculate_risk(x, sigma) for _, x, _ in results_data]
    bars = ax3.barh(algo_names, risks, color=colors, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Risco (Variância)', fontweight='bold')
    ax3.set_title('Comparação de Risco', fontweight='bold', fontsize=12)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, risks):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f' {val:.4f}', va='center', fontweight='bold', fontsize=9)
    
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
                f' ${val:.0f} ({pct:.1f}%)',
                va='center', fontweight='bold', fontsize=8)
    
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
                f' {int(val)} ativos', va='center', fontweight='bold', fontsize=9)
    
    # 6. Sharpe Ratio Comparison (Risk-adjusted return)
    ax6 = fig.add_subplot(gs[1, 2])
    sharpe_ratios = []
    for _, x, _ in results_data:
        ret = calculate_return(x, mu)
        risk = calculate_risk(x, sigma)
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
                f' {val:.3f}', va='center', fontweight='bold', fontsize=9)
    
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
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Dashboard de Comparação Completa dos Algoritmos',
                fontsize=18, fontweight='bold', y=0.998)
    
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
        # ------------------------------------
        
        print("\n(1/5) A executar Hill Climbing...")
        hc_x, hc_fit, hc_hist = hill_climbing(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=HC_MAX_ITER
        )
        
        print("\n(2/5) A executar Simulated Annealing...")
        sa_x, sa_fit, sa_hist = simulated_annealing(
            n, mu, sigma, p, B, k, lambda_val, alpha, beta, max_iter=SA_MAX_ITER
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
        
        print("\n[1/5] Tabela Resumo Estatístico...")
        plot_summary_table(results_data, mu, sigma, p, B, k)
        
        print("\n[2/5] Dashboard de Comparação Completa...")
        plot_algorithm_comparison_dashboard(results_data, mu, sigma, p, B, k)
        
        print("\n[3/5] Gráfico de Convergência Detalhado...")
        plot_convergence([
            ("Hill Climbing", hc_hist),
            ("Simulated Annealing", sa_hist),
            ("Genetic Algorithm", ga_hist),
            ("Tabu Search", ts_hist),
            ("Particle Swarm (PSO)", pso_hist)
        ])
        
        print("\n[4/5] Análise Risco vs Retorno...")
        plot_risk_return_scatter(results_data, mu, sigma)
        
        # --- Gráficos de Alocação (Um conjunto para CADA algoritmo) ---
        print("\n[5/5] Gráficos de Alocação para CADA algoritmo...")
        
        for idx, (name, x, fit) in enumerate(results_data, 1):
            print(f"  [{idx}/{len(results_data)}] Gerando: {name}")
            plot_allocation_charts(x, p, tickers, name, B, k)
        
        total_graphs = 4 + len(results_data) * 3 + 1  # heatmap included
        print("\n" + "="*60)
        print("✓ EXECUÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"✓ Total de gráficos gerados: {total_graphs}")
        print("✓ Resolução: 300 DPI (alta qualidade)")
        print("="*60)
        print("\nGráficos gerados:")
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
