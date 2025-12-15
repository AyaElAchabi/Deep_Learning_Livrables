import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


class GridWorldEnv:
    """
    Environnement GridWorld type Gym pour le Reinforcement Learning
    """
    
    def __init__(self, grid_size=5, goal_reward=10, step_penalty=-0.1, obstacle_penalty=-5, moving_goal=False, goal_move_interval=5):
        """
        Args:
            grid_size: Taille de la grille (grid_size x grid_size)
            goal_reward: Récompense pour atteindre l'objectif
            step_penalty: Pénalité pour chaque pas
            obstacle_penalty: Pénalité pour toucher un obstacle
            moving_goal: Si True, l'objectif se déplace pendant l'épisode
            goal_move_interval: Nombre de pas avant que l'objectif ne se déplace
        """
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        self.moving_goal = moving_goal
        self.goal_move_interval = goal_move_interval
        
        # Positions spéciales
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.obstacles = self._init_obstacles()
        
        # État actuel
        self.agent_pos = None
        self.done = False
        self.step_count = 0
        
        # Actions: 0=haut, 1=droite, 2=bas, 3=gauche
        self.action_space_n = 4
        self.observation_space_n = grid_size * grid_size
        
    def _init_obstacles(self):
        """Initialise quelques obstacles dans la grille"""
        obstacles = []
        if self.grid_size >= 5:
            obstacles = [(1, 1), (2, 2), (3, 1)]
        return obstacles
    
    def _random_goal_position(self):
        """Génère une position aléatoire pour l'objectif"""
        while True:
            goal = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if goal != self.start_pos and goal not in self.obstacles and goal != self.agent_pos:
                return goal
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.agent_pos = self.start_pos
        self.done = False
        self.step_count = 0
        # Placer l'objectif aléatoirement à chaque nouvel épisode
        if self.moving_goal:
            self.goal_pos = self._random_goal_position()
        return self._get_state()
    
    def _get_state(self):
        """Convertit la position en un état unique"""
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def step(self, action):
        """
        Exécute une action
        Args:
            action: 0=haut, 1=droite, 2=bas, 3=gauche
        Returns:
            next_state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {}
        
        # Calculer la nouvelle position
        row, col = self.agent_pos
        
        if action == 0:  # Haut
            row = max(0, row - 1)
        elif action == 1:  # Droite
            col = min(self.grid_size - 1, col + 1)
        elif action == 2:  # Bas
            row = min(self.grid_size - 1, row + 1)
        elif action == 3:  # Gauche
            col = max(0, col - 1)
        
        new_pos = (row, col)
        
        # Calculer la récompense
        if new_pos in self.obstacles:
            reward = self.obstacle_penalty
            # L'agent ne bouge pas s'il touche un obstacle
            new_pos = self.agent_pos
        elif new_pos == self.goal_pos:
            reward = self.goal_reward
            self.done = True
        else:
            reward = self.step_penalty
        
        self.agent_pos = new_pos
        self.step_count += 1
        
        # Déplacer l'objectif pendant l'épisode si activé
        if self.moving_goal and not self.done and self.step_count % self.goal_move_interval == 0:
            old_goal = self.goal_pos
            self.goal_pos = self._random_goal_position()
            # Si l'agent était sur l'ancien objectif, ne pas terminer l'épisode
            if self.agent_pos == old_goal:
                self.done = False
        
        return self._get_state(), reward, self.done, {}
    
    def render(self, ax=None, value_function=None, policy=None):
        """
        Affiche l'environnement avec matplotlib
        Args:
            ax: Axes matplotlib (si None, crée une nouvelle figure)
            value_function: Fonction de valeur à afficher (optionnel)
            policy: Politique à afficher avec des flèches (optionnel)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Grille
        for i in range(self.grid_size + 1):
            ax.axhline(i, color='black', linewidth=0.5)
            ax.axvline(i, color='black', linewidth=0.5)
        
        # Afficher la fonction de valeur si fournie
        if value_function is not None:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = row * self.grid_size + col
                    value = value_function[state]
                    # Normaliser les couleurs
                    color_intensity = (value - value_function.min()) / (value_function.max() - value_function.min() + 1e-10)
                    rect = Rectangle((col, row), 1, 1, facecolor=plt.cm.RdYlGn(color_intensity), alpha=0.6)
                    ax.add_patch(rect)
                    ax.text(col + 0.5, row + 0.5, f'{value:.2f}', 
                           ha='center', va='center', fontsize=8)
        
        # Obstacles
        for obs in self.obstacles:
            rect = Rectangle((obs[1], obs[0]), 1, 1, facecolor='black', alpha=0.7)
            ax.add_patch(rect)
        
        # Position de départ
        start_rect = Rectangle((self.start_pos[1], self.start_pos[0]), 1, 1, 
                               facecolor='blue', alpha=0.3, edgecolor='blue', linewidth=2)
        ax.add_patch(start_rect)
        ax.text(self.start_pos[1] + 0.5, self.start_pos[0] + 0.5, 'S', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Objectif
        goal_rect = Rectangle((self.goal_pos[1], self.goal_pos[0]), 1, 1, 
                             facecolor='green', alpha=0.3, edgecolor='green', linewidth=2)
        ax.add_patch(goal_rect)
        ax.text(self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5, 'G', 
               ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Agent actuel
        if self.agent_pos is not None:
            circle = plt.Circle((self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5), 
                              0.3, color='red', zorder=10)
            ax.add_patch(circle)
        
        # Politique (flèches)
        if policy is not None:
            arrow_map = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if (row, col) not in self.obstacles and (row, col) != self.goal_pos:
                        state = row * self.grid_size + col
                        action = policy[state]
                        dx, dy = arrow_map[action]
                        ax.arrow(col + 0.5, row + 0.5, dx, dy, 
                               head_width=0.15, head_length=0.1, fc='black', ec='black')
        
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_title('GridWorld Environment')
        ax.grid(False)
        
        return ax
