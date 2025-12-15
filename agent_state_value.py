import numpy as np


class StateValueAgent:
    """
    Agent utilisant Value Iteration pour apprendre la fonction de valeur optimale
    """
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Args:
            env: Environnement
            gamma: Facteur de discount
            theta: Seuil de convergence
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space_n
        self.n_actions = env.action_space_n
        
        # Initialiser la fonction de valeur
        self.V = np.zeros(self.n_states)
        # Politique optimale
        self.policy = np.zeros(self.n_states, dtype=int)
    
    def _get_transition_prob(self, state, action):
        """
        Simule une transition pour obtenir le prochain état et la récompense
        Args:
            state: État actuel
            action: Action à prendre
        Returns:
            next_state, reward
        """
        # Convertir l'état en position
        row = state // self.env.grid_size
        col = state % self.env.grid_size
        
        # Sauvegarder l'état actuel de l'environnement
        old_pos = self.env.agent_pos
        old_done = self.env.done
        
        # Simuler l'action
        self.env.agent_pos = (row, col)
        self.env.done = False
        next_state, reward, done, _ = self.env.step(action)
        
        # Restaurer l'état
        self.env.agent_pos = old_pos
        self.env.done = old_done
        
        return next_state, reward, done
    
    def value_iteration(self, max_iterations=1000):
        """
        Algorithme de Value Iteration
        Args:
            max_iterations: Nombre maximum d'itérations
        """
        print("Starting Value Iteration...")
        
        for iteration in range(max_iterations):
            delta = 0
            V_old = self.V.copy()
            
            # Pour chaque état
            for state in range(self.n_states):
                # Convertir en position
                row = state // self.env.grid_size
                col = state % self.env.grid_size
                
                # Ignorer les obstacles et l'objectif
                if (row, col) in self.env.obstacles or (row, col) == self.env.goal_pos:
                    continue
                
                # Calculer la valeur pour chaque action
                action_values = []
                for action in range(self.n_actions):
                    next_state, reward, done = self._get_transition_prob(state, action)
                    value = reward + self.gamma * V_old[next_state] * (1 - done)
                    action_values.append(value)
                
                # Mettre à jour la valeur avec le maximum
                self.V[state] = max(action_values)
                delta = max(delta, abs(V_old[state] - self.V[state]))
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iterations}, Delta: {delta:.6f}")
            
            # Vérifier la convergence
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations!")
                break
        
        # Extraire la politique optimale
        self._extract_policy()
    
    def _extract_policy(self):
        """
        Extrait la politique optimale à partir de la fonction de valeur
        """
        for state in range(self.n_states):
            row = state // self.env.grid_size
            col = state % self.env.grid_size
            
            if (row, col) in self.env.obstacles or (row, col) == self.env.goal_pos:
                continue
            
            action_values = []
            for action in range(self.n_actions):
                next_state, reward, done = self._get_transition_prob(state, action)
                value = reward + self.gamma * self.V[next_state] * (1 - done)
                action_values.append(value)
            
            self.policy[state] = np.argmax(action_values)
    
    def select_action(self, state):
        """
        Sélectionne une action selon la politique optimale
        Args:
            state: État actuel
        Returns:
            Action optimale
        """
        return self.policy[state]
    
    def train(self, max_iterations=1000):
        """
        Entraîne l'agent avec Value Iteration
        Args:
            max_iterations: Nombre maximum d'itérations
        """
        self.value_iteration(max_iterations)
    
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
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {step + 1}")
        
        return episode_rewards
