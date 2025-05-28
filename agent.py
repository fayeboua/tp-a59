from collections import deque            # Structure de mémoire tampon efficace (FIFO)
import random                            # Pour l’échantillonnage aléatoire
import numpy as np                       # Manipulation efficace des vecteurs/tableaux
from model import mlp                    # Fonction qui crée un réseau de neurones MLP


class DQNAgent(object):
    """Agent DQN avec réseau cible pour un apprentissage plus stable"""

    def __init__(self, state_size, action_size):
        # Dimensions de l’environnement
        self.state_size = state_size          # Taille de l'état (ex: nombre de features)
        self.action_size = action_size        # Nombre d’actions possibles (ex: 3 pour buy/hold/sell)

        # Mémoire pour stocker les expériences (état, action, récompense, prochain état, terminal)
        self.memory = deque(maxlen=2000)      # Limite la mémoire à 2000 transitions

        # Hyperparamètres d’apprentissage
        self.gamma = 0.95                     # Facteur de réduction (discount factor)
        self.epsilon = 1.0                    # Taux d’exploration initial (100%)
        self.epsilon_min = 0.01               # Exploration minimale (1%)
        self.epsilon_decay = 0.995            # Décroissance progressive de ε à chaque apprentissage
        self.batch_size = 32                  # Taille du mini-batch pour le replay

        # Réseaux de neurones
        self.model = mlp(state_size, action_size)        # Réseau principal : Q(s, a)
        self.target_model = mlp(state_size, action_size) # Réseau cible : utilisé pour les cibles Q(s', a)
        self.update_target_model()                       # Initialisation du réseau cible

        # Compteur d'étapes d'entraînement
        self.train_step = 0
        self.update_freq = 10                # Fréquence de mise à jour du réseau cible


    def update_target_model(self):
        """Copie les poids du modèle principal vers le réseau cible."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        """Retourne une action selon une politique ε-greedy"""
        if np.random.rand() <= self.epsilon:
            # Explore : retourne une action aléatoire
            return random.randrange(self.action_size)
        # Exploite : prédit Q(s, a) et choisit l’action avec la plus grande valeur
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Retourne l’indice de l’action optimale


    def replay(self, batch_size=None):
        """Entraîne le réseau principal à partir d’un échantillon de mémoire"""
        if batch_size is None:
            batch_size = self.batch_size

        # Ne commence l'entraînement que si on a suffisamment d’expériences
        if len(self.memory) < batch_size:
            return

        # Échantillonnage aléatoire d’un mini-lot
        minibatch = random.sample(self.memory, batch_size)

        # Préparation des tableaux batch (vectorisation)
        states = np.array([t[0][0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3][0] for t in minibatch])
        done = np.array([t[4] for t in minibatch])

        # Prédiction Q(s', a) à partir du réseau cible
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states), axis=1)
        # Si l’épisode est terminé, la récompense est finale
        target[done] = rewards[done]

        # Prédiction Q(s, a) actuelle (réseau principal)
        target_f = self.model.predict(states)

        # Mise à jour des valeurs cibles pour les actions effectuées
        target_f[range(batch_size), actions] = target

        # Entraînement du modèle principal (1 époque, sans affichage)
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Réduction progressive de ε (moins d'exploration avec le temps)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Mise à jour périodique du réseau cible
        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            self.update_target_model()


    def load(self, name):
        """Charge les poids du modèle depuis un fichier"""
        self.model.load_weights(name)
        self.update_target_model()  # Synchronise aussi le modèle cible

    def save(self, name):
        """Sauvegarde les poids du modèle principal dans un fichier"""
        self.model.save_weights(name)

