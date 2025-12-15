import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import GridWorldEnv
from agent_q_learning import QLearningAgent

print("=" * 60)
print("DÉMO: GOAL QUI BOUGE PENDANT L'ÉPISODE")
print("=" * 60)

# Créer environnement avec goal qui bouge PENDANT l'épisode
env = GridWorldEnv(
    grid_size=6,
    goal_reward=10,
    step_penalty=-0.1,
    obstacle_penalty=-5,
    moving_goal=True,
    goal_move_interval=5  # Le goal bouge tous les 5 pas
)

print("\nParamètres:")
print(f"  - Grille: 6x6")
print(f"  - Goal bouge tous les 5 pas")
print(f"  - Entraînement de Q-Learning en cours...")

# Entraîner Q-Learning
q_agent = QLearningAgent(env, alpha=0.15, gamma=0.9, epsilon=0.3)
q_rewards, _ = q_agent.train(num_episodes=1500, max_steps=100, verbose=False)

print(f"  - Récompense moyenne (100 derniers): {np.mean(q_rewards[-100:]):.2f}")
print("\n✅ Agent entraîné!")

# Exécuter un épisode et capturer TOUT
state = env.reset()
trajectory = [env.agent_pos]
goal_positions = [env.goal_pos]
q_values_over_time = [q_agent.get_value_function().copy()]

print(f"\nÉpisode de démonstration:")
print(f"Step 0: Agent={env.agent_pos}, Goal={env.goal_pos}")

for step in range(50):
    action = q_agent.select_action(state, training=False)
    state, reward, done, _ = env.step(action)
    trajectory.append(env.agent_pos)
    goal_positions.append(env.goal_pos)
    q_values_over_time.append(q_agent.get_value_function().copy())
    
    # Afficher quand le goal bouge
    if len(goal_positions) >= 2 and goal_positions[-1] != goal_positions[-2]:
        print(f"Step {step+1}: 🎯 GOAL BOUGE! Nouvelle position: {env.goal_pos}")
    
    if done:
        print(f"Step {step+1}: ✅ GOAL ATTEINT! Agent={env.agent_pos}")
        break

print(f"\nTotal steps: {len(trajectory)-1}")
print(f"Goal a bougé {len(set(goal_positions))-1} fois")

# Créer animation avec Q-values qui s'adaptent
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

def init():
    env.agent_pos = trajectory[0]
    env.goal_pos = goal_positions[0]
    
    ax1.clear()
    env.render(ax=ax1)
    ax1.set_title(f'Environnement - Step 0/{len(trajectory)-1}')
    
    ax2.clear()
    env.render(ax=ax2, value_function=q_values_over_time[0])
    ax2.set_title(f'Q-Values - Goal: {goal_positions[0]}')
    
    return ax1, ax2

def animate_frame(frame):
    env.agent_pos = trajectory[frame]
    env.goal_pos = goal_positions[frame]
    
    # Graphique 1: Environnement avec trajectoire
    ax1.clear()
    env.render(ax=ax1)
    for i in range(frame):
        start = trajectory[i]
        end = trajectory[i + 1]
        ax1.plot([start[1] + 0.5, end[1] + 0.5], 
               [start[0] + 0.5, end[0] + 0.5], 
               color='purple', linewidth=2, alpha=0.7)
    ax1.set_title(f'Agent Q-Learning - Step {frame}/{len(trajectory)-1}')
    
    # Graphique 2: Q-Values mis à jour (montrent l'adaptation au goal mobile)
    ax2.clear()
    env.render(ax=ax2, value_function=q_values_over_time[frame])
    ax2.set_title(f'Q-Values adaptées - Goal: {goal_positions[frame]}')
    
    return ax1, ax2

print("\n🎬 Création de l'animation...")
anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                               frames=len(trajectory), interval=400,
                               blit=False, repeat=True)

print("✅ Animation prête!")
print("\nREGARDEZ:")
print("  - À GAUCHE: L'agent (cercle rouge) poursuit le goal mobile")
print("  - À DROITE: Les Q-values s'adaptent quand le goal bouge")
print("  - Le goal (carré vert G) change de position tous les 5 pas")

plt.tight_layout()
plt.show()
