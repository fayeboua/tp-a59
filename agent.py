from collections import deque      # Structure de données optimisée pour la mémoire tampon
import random                      # Pour sélectionner des échantillons aléatoires
import numpy as np                 # Manipulation efficace des vecteurs/tableaux
from model import mlp              # Fonction qui retourne un réseau de neurones (MLP = Multi-Layer Perceptron)




class DQNAgent(object):
  """ Une classe représentant un agent DQN simple, capable d'apprendre à partir de transitions (état, action, récompense, nouvel état) """
  def __init__(self, state_size, action_size):   # Initialisation de l’agent
    self.state_size = state_size                 # Taille de l’espace d’états
    self.action_size = action_size               # Nombre d’actions possibles
    self.memory = deque(maxlen=2000)             # Mémoire des transitions (buffer d’expérience)
    self.gamma = 0.95                            # Taux d’actualisation (discount factor)
    self.epsilon = 1.0                           # Taux d’exploration initial (ε-greedy)
    self.epsilon_min = 0.01                      # Valeur minimale de ε
    self.epsilon_decay = 0.995                   # Facteur de décroissance de ε après chaque replay
    self.model = mlp(state_size, action_size)    # Réseau de neurones Q(s, a)


  # Stockage d’une expérience dans la mémoire
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  # Choix d’une action (exploration/exploitation) (policy ε-greedy)
  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)       # Explore : action aléatoire
    act_values = self.model.predict(state)            # Exploite : Q(s, a) → argmax a
    return np.argmax(act_values[0])                   # returne action


  # Entraînement à partir de la mémoire (replay)
  def replay(self, batch_size=32):
    """ vectorized implementation; 30x speed up compared with for loop """
    minibatch = random.sample(self.memory, batch_size)

    # Échantillonne un mini-lot aléatoire dans la mémoire.
    states = np.array([tup[0][0] for tup in minibatch])
    actions = np.array([tup[1] for tup in minibatch])
    rewards = np.array([tup[2] for tup in minibatch])
    next_states = np.array([tup[3][0] for tup in minibatch])
    done = np.array([tup[4] for tup in minibatch])

    # Q(s', a)
    target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]

    # Q(s, a)
    target_f = self.model.predict(states)
    # make the agent to approximately map the current state to future discounted reward
    target_f[range(batch_size), actions] = target

    # Met à jour la sortie du réseau pour les actions prises, en gardant les autres inchangées
    self.model.fit(states, target_f, epochs=1, verbose=0)

    # Entraîne le réseau sur le batch, une seule époque.
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  # Chargement et sauvegarde du modèle
  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)