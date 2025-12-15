# Mini Projet Reinforcement Learning

Projet de reinforcement learning avec environnement GridWorld dynamique, agent random, agent state value et **agent Q-Learning**.

## Structure du Projet

- `environment.py` - Environnement GridWorld type OpenAI Gym avec **objectif mobile**
- `agent_random.py` - Agent qui choisit des actions aléatoires
- `agent_state_value.py` - Agent utilisant Value Iteration (pour environnements statiques)
- `agent_q_learning.py` - **Agent utilisant Q-Learning (adapté aux environnements dynamiques)**
- `main.py` - Script principal pour entraîner et comparer les agents

## Nouveautés

### 🎯 Objectif Mobile
L'environnement peut maintenant avoir un objectif qui se déplace :
- **À chaque épisode** : nouvelle position aléatoire
- **Pendant l'épisode** : changement toutes les N steps (configurable)

### 🧠 Agent Q-Learning
- Apprend la fonction Q optimale Q(s,a)
- Exploration epsilon-greedy avec décroissance
- S'adapte aux environnements dynamiques
- Convergence progressive vers la politique optimale

## Installation

```bash
pip install numpy matplotlib
```

## Utilisation

Exécutez simplement le script principal :

```bash
python main.py
```

## Configuration

Dans `main.py`, vous pouvez modifier les paramètres suivants :

### Paramètres de l'environnement
- `GRID_SIZE` : Taille de la grille (par défaut : 5)
- `GOAL_REWARD` : Récompense pour atteindre l'objectif (par défaut : 10)
- `STEP_PENALTY` : Pénalité pour chaque pas (par défaut : -0.1)
- `OBSTACLE_PENALTY` : Pénalité pour toucher un obstacle (par défaut : -5)
- `MOVING_GOAL` : **Activer l'objectif mobile** (par défaut : True)
- `GOAL_MOVE_INTERVAL` : **Nombre de pas avant déplacement de l'objectif** (par défaut : 10)

### Paramètres d'apprentissage
- `GAMMA` : Facteur de discount (par défaut : 0.9)
- `ALPHA` : **Taux d'apprentissage pour Q-Learning** (par défaut : 0.1)
- `EPSILON` : **Taux d'exploration initial pour Q-Learning** (par défaut : 0.3)
- `NUM_EPISODES_RANDOM` : Nombre d'épisodes pour l'agent random (par défaut : 100)
- `NUM_EPISODES_Q_LEARNING` : **Nombre d'épisodes pour Q-Learning** (par défaut : 1000)
- `NUM_EPISODES_EVAL` : Nombre d'épisodes pour l'évaluation (par défaut : 10)
- `MAX_STEPS` : Nombre maximum de pas par épisode (par défaut : 100)

## Fonctionnalités

### Environnement GridWorld Dynamique
- Grille personnalisable
- Obstacles statiques
- Position de départ (S) fixe
- **Objectif (G) mobile** :
  - Nouvelle position aléatoire à chaque épisode
  - Peut se déplacer pendant l'épisode
- Récompenses configurables
- Visualisation avec matplotlib

### Agent Random
- Sélection d'actions aléatoires
- Utilisé comme baseline de comparaison

### Agent Q-Learning ⭐
- **Algorithme** : Q-Learning (temporal difference)
- **Table Q** : Stocke Q(s,a) pour chaque paire état-action
- **Politique** : Epsilon-greedy avec décroissance
- **Update Rule** : Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Avantages** :
  - S'adapte aux environnements dynamiques
  - Pas besoin de modèle de l'environnement
  - Apprentissage en ligne

### Agent State Value
- Algorithme : Value Iteration (programmation dynamique)
- Apprend la fonction de valeur optimale V*(s)
- Extrait une politique optimale
- **Note** : Fonctionne mieux sur environnements statiques
- Converge vers la solution optimale

## Visualisation

Le programme génère plusieurs graphiques :

1. **environment_visualization.png** : 
   - Environnement de base
   - Fonction Q et politique Q-Learning (flèches)
   - Trajectoire optimale Q-Learning
   - [Fonction V et trajectoire State Value si objectif statique]

2. **training_results.png** :
   - Performance de l'agent random
   - Courbe d'apprentissage Q-Learning
   - [Évaluation State Value si objectif statique]

3. **Animations interactives** :
   - Mouvement en temps réel de Q-Learning
   - Mouvement en temps réel de Random
   - [Mouvement de State Value si objectif statique]

## Comparaison des Algorithmes

| Algorithme | Type | Environnement | Convergence | Complexité |
|------------|------|---------------|-------------|------------|
| **Q-Learning** | Model-free TD | Statique/Dynamique | Progressive | O(S×A) |
| **Value Iteration** | Model-based DP | Statique | Rapide | O(S²×A) |
| **Random** | Baseline | Tous | Aucune | O(1) |

## Actions

- 0 : Haut ↑
- 1 : Droite →
- 2 : Bas ↓
- 3 : Gauche ←

## Exemples de Configuration

### Environnement Dynamique (par défaut)
```python
MOVING_GOAL = True
GOAL_MOVE_INTERVAL = 10
NUM_EPISODES_Q_LEARNING = 1000
ALPHA = 0.1
EPSILON = 0.3
```

### Environnement Statique
```python
MOVING_GOAL = False
# L'agent State Value sera également entraîné
```

### Grille Plus Grande
```python
GRID_SIZE = 10
GOAL_REWARD = 20
STEP_PENALTY = -0.5
OBSTACLE_PENALTY = -10
```

## Résultats Attendus

- **Agent Random** : Performance erratique, pas d'apprentissage
- **Agent Q-Learning** : Amélioration progressive, convergence vers politique optimale
- **Agent State Value** : Performance optimale dès l'entraînement (si statique)

L'agent Q-Learning devrait significativement surpasser l'agent Random, surtout après plusieurs épisodes d'entraînement. Dans un environnement statique, State Value converge plus rapidement mais Q-Learning finit par atteindre une performance similaire.
