import numpy as np


class RandomAgent:
    """
    Agent qui choisit des actions aléatoires
    """
    
    def __init__(self, action_space_n):
        """
        Args:
            action_space_n: Nombre d'actions possibles
        """
        self.action_space_n = action_space_n
    
    def select_action(self, state):
        """
        Sélectionne une action aléatoire
        Args:
            state: État actuel (non utilisé pour un agent random)
        Returns:
            Action aléatoire
        """
        return np.random.randint(0, self.action_space_n)
    
    def train(self, env, num_episodes=100, max_steps=100):
        """
        Exécute des épisodes avec des actions aléatoires
        Args:
            env: Environnement
            num_episodes: Nombre d'épisodes
            max_steps: Nombre maximum de pas par épisode
        Returns:
            Liste des récompenses totales par épisode
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward (last 10): {avg_reward:.2f}")
        
        return episode_rewards
