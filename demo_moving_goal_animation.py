import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import GridWorldEnv
from agent_q_learning import QLearningAgent

# Créer environnement avec moving goal
env = GridWorldEnv(
    grid_size=5,
    goal_reward=10,
    step_penalty=-0.1,
    obstacle_penalty=-5,
    moving_goal=True,
    goal_move_interval=5  # Goal bouge tous les 5 pas
)

# Créer un agent simple
q_agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
print("Entraînement rapide de l'agent...")
q_agent.train(num_episodes=500, max_steps=100, verbose=False)

# Créer animation
fig, ax = plt.subplots(figsize=(8, 8))

state = env.reset()
trajectory = [env.agent_pos]
goal_positions = [env.goal_pos]

print(f"\nDémonstration avec goal mobile (bouge tous les 5 pas):")
print(f"Step 0: Agent en {env.agent_pos}, Goal en {env.goal_pos}")

for step in range(30):
    action = q_agent.select_action(state, training=False)
    state, reward, done, _ = env.step(action)
    trajectory.append(env.agent_pos)
    goal_positions.append(env.goal_pos)
    
    if (step + 1) % 5 == 0 or done:
        print(f"Step {step+1}: Agent en {env.agent_pos}, Goal en {env.goal_pos}")
    
    if done:
        print(f"🎯 GOAL ATTEINT à step {step+1}!")
        break

def init():
    env.agent_pos = trajectory[0]
    env.goal_pos = goal_positions[0]
    env.render(ax=ax)
    ax.set_title(f'Step 0/{len(trajectory)-1} | Goal: {goal_positions[0]}')
    return ax,

def animate_frame(frame):
    ax.clear()
    env.agent_pos = trajectory[frame]
    env.goal_pos = goal_positions[frame]
    env.render(ax=ax)
    
    # Tracer trajectoire
    for i in range(frame):
        start = trajectory[i]
        end = trajectory[i + 1]
        ax.plot([start[1] + 0.5, end[1] + 0.5], 
               [start[0] + 0.5, end[0] + 0.5], 
               color='purple', linewidth=2, alpha=0.7)
    
    ax.set_title(f'Q-Learning Agent - Step {frame}/{len(trajectory)-1} | Goal: {goal_positions[frame]}')
    return ax,

anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                               frames=len(trajectory), interval=300,
                               blit=False, repeat=True)

print("\n✅ Animation créée - vous pouvez voir le goal (carré vert) se déplacer!")
plt.show()
