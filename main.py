import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import GridWorldEnv
from agent_random import RandomAgent
from agent_state_value import StateValueAgent
from agent_q_learning import QLearningAgent


def plot_training_results(random_rewards, sv_rewards, q_rewards):
    """
    Affiche les résultats d'entraînement
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Agent Random
    ax1.plot(random_rewards, alpha=0.6, label='Reward par épisode')
    window = 10
    if len(random_rewards) >= window:
        moving_avg = np.convolve(random_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(random_rewards)), moving_avg, 
                linewidth=2, label=f'Moyenne mobile ({window})')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Agent Random - Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agent State Value
    if sv_rewards:
        ax2.bar(range(len(sv_rewards)), sv_rewards, alpha=0.7, color='green')
        ax2.axhline(y=np.mean(sv_rewards), color='r', linestyle='--', 
                   linewidth=2, label=f'Moyenne: {np.mean(sv_rewards):.2f}')
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Récompense totale')
        ax2.set_title('Agent State Value - Évaluation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Agent Q-Learning
    if q_rewards:
        ax3.plot(q_rewards, alpha=0.6, label='Reward par épisode', color='purple')
        window = 50
        if len(q_rewards) >= window:
            moving_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(q_rewards)), moving_avg, 
                    linewidth=2, label=f'Moyenne mobile ({window})', color='red')
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Récompense totale')
        ax3.set_title('Agent Q-Learning - Entraînement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:/Users/pc/Desktop/RL/training_results.png', dpi=150)
    print("\nGraphique sauvegardé: training_results.png")


def animate_agent(env, agent, agent_name, max_steps=100, interval=500):
    """
    Anime les mouvements de l'agent dans l'environnement
    Args:
        env: Environnement
        agent: Agent à animer
        agent_name: Nom de l'agent pour le titre
        max_steps: Nombre maximum de pas
        interval: Délai entre les frames en ms
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Collecter la trajectoire ET les positions du goal
    state = env.reset()
    trajectory = [env.agent_pos]
    goal_positions = [env.goal_pos]  # Capturer la position du goal à chaque step
    states = [state]
    
    for _ in range(max_steps):
        if hasattr(agent, 'select_action'):
            # Q-Learning ou Random agent
            if hasattr(agent, 'Q'):  # Q-Learning
                action = agent.select_action(state, training=False)
            else:  # Random
                action = agent.select_action(state)
        else:
            action = agent.select_action(state)
        
        state, _, done, _ = env.step(action)
        trajectory.append(env.agent_pos)
        goal_positions.append(env.goal_pos)  # Capturer la nouvelle position du goal
        states.append(state)
        if done:
            break
    
    # Fonction d'initialisation
    def init():
        env.agent_pos = trajectory[0]
        env.goal_pos = goal_positions[0]
        if hasattr(agent, 'V'):  # State Value
            env.render(ax=ax, value_function=agent.V)
        elif hasattr(agent, 'Q'):  # Q-Learning
            value_func = agent.get_value_function()
            env.render(ax=ax, value_function=value_func)
        else:
            env.render(ax=ax)
        ax.set_title(f'{agent_name} - Step 0/{len(trajectory)-1}')
        return ax,
    
    # Fonction d'animation
    def animate(frame):
        ax.clear()
        env.agent_pos = trajectory[frame]
        env.goal_pos = goal_positions[frame]  # Mettre à jour la position du goal
        
        if hasattr(agent, 'V'):  # State Value
            env.render(ax=ax, value_function=agent.V)
            color = 'red'
        elif hasattr(agent, 'Q'):  # Q-Learning
            value_func = agent.get_value_function()
            env.render(ax=ax, value_function=value_func)
            color = 'purple'
        else:
            env.render(ax=ax)
            color = 'orange'
        
        # Tracer la trajectoire jusqu'à maintenant
        for i in range(frame):
            start = trajectory[i]
            end = trajectory[i + 1]
            ax.plot([start[1] + 0.5, end[1] + 0.5], 
                   [start[0] + 0.5, end[0] + 0.5], 
                   color=color, linewidth=2, alpha=0.7)
        
        # Afficher si le goal a bougé
        goal_text = f'Goal: {goal_positions[frame]}'
        ax.set_title(f'{agent_name} - Step {frame}/{len(trajectory)-1} | {goal_text}')
        return ax,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(trajectory), interval=interval,
                                   blit=False, repeat=True)
    
    return fig, anim


def main():
    # ========== PARAMÈTRES MODIFIABLES ==========
    GRID_SIZE = 5
    GOAL_REWARD = 10
    STEP_PENALTY = -0.1
    OBSTACLE_PENALTY = -5
    MOVING_GOAL = True  # Activer le déplacement de l'objectif
    GOAL_MOVE_INTERVAL = 10  # L'objectif se déplace tous les 10 pas
    
    GAMMA = 0.9  # Facteur de discount
    ALPHA = 0.1  # Taux d'apprentissage pour Q-Learning
    EPSILON = 0.3  # Taux d'exploration initial pour Q-Learning
    
    NUM_EPISODES_RANDOM = 100
    NUM_EPISODES_Q_LEARNING = 1000
    NUM_EPISODES_EVAL = 10
    MAX_STEPS = 100
    # ============================================
    
    print("=" * 60)
    print("MINI PROJET REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"\nParamètres:")
    print(f"  - Taille de grille: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  - Récompense objectif: {GOAL_REWARD}")
    print(f"  - Pénalité par pas: {STEP_PENALTY}")
    print(f"  - Pénalité obstacle: {OBSTACLE_PENALTY}")
    print(f"  - Objectif mobile: {MOVING_GOAL}")
    if MOVING_GOAL:
        print(f"  - Intervalle de déplacement: {GOAL_MOVE_INTERVAL} pas")
    print(f"  - Facteur de discount (gamma): {GAMMA}")
    print(f"  - Taux d'apprentissage (alpha): {ALPHA}")
    print(f"  - Taux d'exploration initial (epsilon): {EPSILON}")
    
    # Créer l'environnement
    env = GridWorldEnv(
        grid_size=GRID_SIZE,
        goal_reward=GOAL_REWARD,
        step_penalty=STEP_PENALTY,
        obstacle_penalty=OBSTACLE_PENALTY,
        moving_goal=MOVING_GOAL,
        goal_move_interval=GOAL_MOVE_INTERVAL
    )
    
    # ========== AGENT RANDOM ==========
    print("\n" + "=" * 60)
    print("1. ENTRAÎNEMENT AGENT RANDOM")
    print("=" * 60)
    
    random_agent = RandomAgent(env.action_space_n)
    random_rewards = random_agent.train(env, num_episodes=NUM_EPISODES_RANDOM, max_steps=MAX_STEPS)
    
    print(f"\nRésultat Agent Random:")
    print(f"  - Récompense moyenne: {np.mean(random_rewards):.2f}")
    print(f"  - Récompense médiane: {np.median(random_rewards):.2f}")
    print(f"  - Meilleure récompense: {np.max(random_rewards):.2f}")
    
    # ========== AGENT Q-LEARNING ==========
    print("\n" + "=" * 60)
    print("2. ENTRAÎNEMENT AGENT Q-LEARNING")
    print("=" * 60)
    
    q_agent = QLearningAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    q_rewards, q_steps = q_agent.train(num_episodes=NUM_EPISODES_Q_LEARNING, max_steps=MAX_STEPS)
    
    print(f"\nRésultat Agent Q-Learning (entraînement):")
    print(f"  - Récompense moyenne (100 derniers): {np.mean(q_rewards[-100:]):.2f}")
    print(f"  - Pas moyens (100 derniers): {np.mean(q_steps[-100:]):.1f}")
    
    print("\n" + "=" * 60)
    print("3. ÉVALUATION AGENT Q-LEARNING")
    print("=" * 60)
    
    q_eval_rewards = q_agent.evaluate(num_episodes=NUM_EPISODES_EVAL, max_steps=MAX_STEPS)
    
    print(f"\nRésultat Agent Q-Learning (évaluation):")
    print(f"  - Récompense moyenne: {np.mean(q_eval_rewards):.2f}")
    print(f"  - Récompense médiane: {np.median(q_eval_rewards):.2f}")
    print(f"  - Meilleure récompense: {np.max(q_eval_rewards):.2f}")
    
    # ========== AGENT STATE VALUE (optionnel pour grilles statiques) ==========
    sv_agent = None
    sv_rewards = []
    
    if not MOVING_GOAL:
        print("\n" + "=" * 60)
        print("4. ENTRAÎNEMENT AGENT STATE VALUE")
        print("=" * 60)
        
        sv_agent = StateValueAgent(env, gamma=GAMMA)
        sv_agent.train(max_iterations=1000)
        
        print("\n" + "=" * 60)
        print("5. ÉVALUATION AGENT STATE VALUE")
        print("=" * 60)
        
        sv_rewards = sv_agent.evaluate(num_episodes=NUM_EPISODES_EVAL, max_steps=MAX_STEPS)
        
        print(f"\nRésultat Agent State Value:")
        print(f"  - Récompense moyenne: {np.mean(sv_rewards):.2f}")
        print(f"  - Récompense médiane: {np.median(sv_rewards):.2f}")
        print(f"  - Meilleure récompense: {np.max(sv_rewards):.2f}")
    
    # ========== VISUALISATION ==========
    print("\n" + "=" * 60)
    print("VISUALISATION")
    print("=" * 60)
    
    # Créer une figure simple avec 2 graphiques
    fig = plt.figure(figsize=(12, 6))
    
    # Environnement de base
    ax1 = plt.subplot(1, 2, 1)
    env.reset()
    env.render(ax=ax1)
    ax1.set_title('Environnement GridWorld')
    
    # Q-Learning: Fonction de valeur et politique
    ax2 = plt.subplot(1, 2, 2)
    env.reset()
    q_value_func = q_agent.get_value_function()
    q_policy = q_agent.get_policy()
    env.render(ax=ax2, value_function=q_value_func, policy=q_policy)
    ax2.set_title('Q-Learning: Fonction Q + Politique')
    
    plt.tight_layout()
    plt.savefig('c:/Users/pc/Desktop/RL/environment_visualization.png', dpi=150)
    print("Graphique sauvegardé: environment_visualization.png")
    plt.show()
    plt.close()
    
    # Graphiques de performance
    plot_training_results(random_rewards, sv_rewards, q_rewards)
    plt.show()
    plt.close()
    
    # ========== ANIMATIONS ==========
    print("\n" + "=" * 60)
    print("ANIMATIONS DES AGENTS")
    print("=" * 60)
    
    print("\n1. Animation de l'agent Q-Learning...")
    input("Appuyez sur Entrée pour démarrer l'animation Q-Learning...")
    fig_anim_q, anim_q = animate_agent(env, q_agent, "Agent Q-Learning", 
                                       max_steps=MAX_STEPS, interval=500)
    plt.show()
    plt.close()
    
    print("\n2. Animation de l'agent Random...")
    input("Appuyez sur Entrée pour démarrer l'animation Random...")
    fig_anim_random, anim_random = animate_agent(env, random_agent, "Agent Random", 
                                                  max_steps=MAX_STEPS, interval=500)
    plt.show()
    plt.close()
    
    print("\n" + "=" * 60)
    print("COMPARAISON FINALE")
    print("=" * 60)
    print(f"Agent Random - Récompense moyenne: {np.mean(random_rewards):.2f}")
    print(f"Agent Q-Learning - Récompense moyenne: {np.mean(q_eval_rewards):.2f}")
    if sv_rewards:
        print(f"Agent State Value - Récompense moyenne: {np.mean(sv_rewards):.2f}")
    
    if sv_rewards:
        print(f"\nAmélioration Q-Learning vs Random: {((np.mean(q_eval_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")
        print(f"Amélioration State Value vs Random: {((np.mean(sv_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")
    else:
        print(f"\nAmélioration Q-Learning vs Random: {((np.mean(q_eval_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")


if __name__ == "__main__":
    main()
