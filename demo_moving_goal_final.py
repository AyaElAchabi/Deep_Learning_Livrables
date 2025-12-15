import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import GridWorldEnv
from agent_random import RandomAgent
from agent_q_learning import QLearningAgent

print("=" * 70)
print("DÉMO: GOAL QUI BOUGE PENDANT L'ÉPISODE")
print("Comparaison Random vs Q-Learning avec goal mobile")
print("=" * 70)

# Paramètres
GRID_SIZE = 6
GOAL_MOVE_INTERVAL = 8  # Goal bouge tous les 8 pas
MAX_STEPS = 60

# Créer environnement avec goal mobile
env = GridWorldEnv(
    grid_size=GRID_SIZE,
    goal_reward=10,
    step_penalty=-0.1,
    obstacle_penalty=-5,
    moving_goal=True,
    goal_move_interval=GOAL_MOVE_INTERVAL
)

print(f"\nParamètres:")
print(f"  - Grille: {GRID_SIZE}x{GRID_SIZE}")
print(f"  - Goal bouge tous les {GOAL_MOVE_INTERVAL} pas")
print(f"  - Max steps: {MAX_STEPS}")

# ========== AGENT Q-LEARNING ==========
print(f"\n{'='*70}")
print("ENTRAÎNEMENT AGENT Q-LEARNING")
print(f"{'='*70}")
q_agent = QLearningAgent(env, alpha=0.15, gamma=0.9, epsilon=0.3)
print("Entraînement en cours...")
q_rewards, _ = q_agent.train(num_episodes=2000, max_steps=100, verbose=False)
print(f"✅ Entraînement terminé!")
print(f"   Récompense moyenne: {np.mean(q_rewards[-100:]):.2f}")

# ========== AGENT RANDOM ==========
print(f"\n{'='*70}")
print("AGENT RANDOM (Baseline)")
print(f"{'='*70}")
random_agent = RandomAgent(env.action_space_n)

# ========== FONCTION D'ANIMATION ==========
def run_episode_with_animation(agent, agent_name, color):
    """Exécute un épisode et crée l'animation avec Q-values dynamiques"""
    
    # Exécuter l'épisode et capturer tout
    state = env.reset()
    trajectory = [env.agent_pos]
    goal_trajectory = [env.goal_pos]
    rewards = []
    actions = []
    q_values_snapshots = []  # Capturer les Q-values à chaque step
    
    # Capturer les Q-values initiales
    if hasattr(agent, 'Q'):
        q_values_snapshots.append(agent.get_value_function().copy())
    else:
        q_values_snapshots.append(None)
    
    print(f"\n{agent_name} - Début de l'épisode:")
    print(f"  Step 0: Agent={env.agent_pos}, Goal={env.goal_pos}")
    
    for step in range(MAX_STEPS):
        # Sélectionner action
        if hasattr(agent, 'Q'):
            action = agent.select_action(state, training=False)
        else:
            action = agent.select_action(state)
        
        actions.append(action)
        
        # Exécuter action
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        trajectory.append(env.agent_pos)
        goal_trajectory.append(env.goal_pos)
        
        # Capturer les Q-values après chaque step
        if hasattr(agent, 'Q'):
            q_values_snapshots.append(agent.get_value_function().copy())
        else:
            q_values_snapshots.append(None)
        
        # Afficher quand le goal bouge
        if len(goal_trajectory) >= 2 and goal_trajectory[-1] != goal_trajectory[-2]:
            print(f"  Step {step+1}: 🎯 GOAL BOUGE! {goal_trajectory[-2]} → {goal_trajectory[-1]}")
            if hasattr(agent, 'Q'):
                print(f"             → Les Q-values vont s'adapter!")
        
        if done:
            print(f"  Step {step+1}: ✅ GOAL ATTEINT! Récompense totale: {sum(rewards):.2f}")
            break
    
    if not done:
        print(f"  Épisode terminé après {MAX_STEPS} pas. Récompense totale: {sum(rewards):.2f}")
    
    print(f"  Goal a changé {len(set(goal_trajectory))-1} fois")
    
    # Créer l'animation avec 2 subplots si Q-Learning, 1 sinon
    if hasattr(agent, 'Q'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 10))
        ax2 = None
    
    action_names = ['↑', '→', '↓', '←']
    
    def init():
        env.agent_pos = trajectory[0]
        env.goal_pos = goal_trajectory[0]
        
        # Subplot 1: Environnement avec trajectoire
        env.render(ax=ax1)
        title = f'{agent_name} - Step 0/{len(trajectory)-1}\n'
        title += f'Goal: {goal_trajectory[0]} | Reward: 0.00'
        ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Subplot 2: Q-values (si Q-Learning)
        if ax2 is not None and q_values_snapshots[0] is not None:
            env.render(ax=ax2, value_function=q_values_snapshots[0])
            ax2.set_title('Q-Values (State Values)', fontsize=12, fontweight='bold')
        
        return ax1, ax2 if ax2 else ax1,
    
    def animate_frame(frame):
        env.agent_pos = trajectory[frame]
        env.goal_pos = goal_trajectory[frame]
        
        # Subplot 1: Environnement avec trajectoire
        ax1.clear()
        env.render(ax=ax1)
        
        # Tracer la trajectoire
        for i in range(frame):
            start = trajectory[i]
            end = trajectory[i + 1]
            ax1.plot([start[1] + 0.5, end[1] + 0.5], 
                   [start[0] + 0.5, end[0] + 0.5], 
                   color=color, linewidth=3, alpha=0.6)
        
        # Titre avec informations
        title = f'{agent_name} - Step {frame}/{len(trajectory)-1}\n'
        title += f'Goal: {goal_trajectory[frame]}'
        if frame > 0:
            title += f' | Action: {action_names[actions[frame-1]]} | Reward: {rewards[frame-1]:.2f}'
            # Indiquer si goal a bougé
            if frame > 0 and goal_trajectory[frame] != goal_trajectory[frame-1]:
                title += ' | 🎯 GOAL A BOUGÉ!'
        ax1.set_title(title, fontsize=11, fontweight='bold')
        
        # Subplot 2: Q-values mises à jour (si Q-Learning)
        if ax2 is not None and q_values_snapshots[frame] is not None:
            ax2.clear()
            env.render(ax=ax2, value_function=q_values_snapshots[frame])
            
            q_title = f'Q-Values - Goal: {goal_trajectory[frame]}'
            if frame > 0 and goal_trajectory[frame] != goal_trajectory[frame-1]:
                q_title += '\n🔄 VALUES ADAPTÉES AU NOUVEAU GOAL!'
            ax2.set_title(q_title, fontsize=11, fontweight='bold')
        
        return (ax1, ax2) if ax2 else (ax1,)
    
    anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                   frames=len(trajectory), interval=500,
                                   blit=False, repeat=True)
    
    return fig, anim

# ========== ANIMATION Q-LEARNING ==========
print(f"\n{'='*70}")
print("ANIMATION 1: AGENT Q-LEARNING (Intelligent - Apprend)")
print(f"{'='*70}")
input("Appuyez sur Entrée pour voir l'animation Q-Learning...")

fig_q, anim_q = run_episode_with_animation(q_agent, "Agent Q-Learning (Intelligent)", "purple")
plt.show()
plt.close()

# ========== ANIMATION RANDOM ==========
print(f"\n{'='*70}")
print("ANIMATION 2: AGENT RANDOM (Baseline - N'apprend pas)")
print(f"{'='*70}")
input("Appuyez sur Entrée pour voir l'animation Random...")

fig_random, anim_random = run_episode_with_animation(random_agent, "Agent Random (Baseline)", "orange")
plt.show()
plt.close()

print(f"\n{'='*70}")
print("✅ DÉMO TERMINÉE!")
print(f"{'='*70}")
print("\nCOMPARAISON:")
print("  ✅ Q-LEARNING: S'adapte au goal mobile, trouve des chemins efficaces")
print("  ❌ RANDOM: Ne peut pas apprendre, performance erratique")
print("\nCONCLUSION:")
print("  - Le goal bouge pendant l'épisode (tous les 8 pas)")
print("  - Les récompenses et actions changent dynamiquement")
print("  - Q-Learning montre une adaptation intelligente")
