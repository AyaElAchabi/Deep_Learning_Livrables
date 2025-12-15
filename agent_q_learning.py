import numpy as np


class QLearningAgent:
    """
    Agent utilisant Q-Learning pour apprendre la fonction Q optimale
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Taux d'exploration (epsilon-greedy)
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.n_states = env.observation_space_n
        self.n_actions = env.action_space_n
        
        # Initialiser la table Q
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def select_action(self, state, training=True):
        """
        Sélectionne une action selon epsilon-greedy
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy; sinon prend la meilleure action
        Returns:
            Action sélectionnée
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: meilleure action selon Q
            return np.argmax(self.Q[state])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-value en utilisant l'équation de Q-Learning
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
        """
        # Q-Learning update rule
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action] * (1 - done)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def train(self, num_episodes=1000, max_steps=100, verbose=True):
        """
        Entraîne l'agent avec Q-Learning
        Args:
            num_episodes: Nombre d'épisodes
            max_steps: Nombre maximum de pas par épisode
            verbose: Si True, affiche les progrès
        Returns:
            Liste des récompenses totales par épisode
        """
        episode_rewards = []
        episode_steps = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Sélectionner et exécuter une action
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                
                # Mettre à jour Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)
            
            # Décroissance d'epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_steps = np.mean(episode_steps[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Steps: {avg_steps:.1f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards, episode_steps
    
    def evaluate(self, num_episodes=10, max_steps=100):
        """
        Évalue la politique apprise
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre maximum de pas par épisode
        Returns:
            Liste des récompenses totales par épisode
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state, training=False)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {step + 1}")
        
        return episode_rewards
    
    def get_policy(self):
        """
        Extrait la politique à partir de la table Q
        Returns:
            Politique (array d'actions pour chaque état)
        """
        return np.argmax(self.Q, axis=1)
    
    def get_value_function(self):
        """
        Extrait la fonction de valeur à partir de la table Q
        Returns:
            Fonction de valeur (max Q-value pour chaque état)
        """
        return np.max(self.Q, axis=1)
