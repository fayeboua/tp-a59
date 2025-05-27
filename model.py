# Importation des composants nécessaires de Keras (version intégrée à TensorFlow)
from tensorflow.keras.models import Sequential              # Pour créer un modèle de réseau de neurones séquentiel
from tensorflow.keras.layers import Dense                   # Pour ajouter des couches entièrement connectées (fully connected)
from tensorflow.keras.optimizers import Adam                # Optimiseur Adam pour l'entraînement

def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse'):
    """
    Crée un perceptron multicouche (MLP) pour des tâches d'apprentissage supervisé ou par renforcement.
    
    Paramètres :
    - n_obs : int, nombre d'observations (dimensions d'entrée)
    - n_action : int, nombre de sorties (par exemple, actions possibles)
    - n_hidden_layer : int, nombre de couches cachées (par défaut 1)
    - n_neuron_per_layer : int, nombre de neurones par couche cachée (par défaut 32)
    - activation : str, fonction d'activation à utiliser dans les couches cachées (par défaut 'relu')
    - loss : str, fonction de perte (par défaut 'mse' — mean squared error)

    Retour :
    - model : objet Keras, modèle compilé prêt à entraîner
    """

    # Création d'un modèle séquentiel (empilement linéaire de couches)
    model = Sequential()

    # Ajout de la première couche dense avec la forme d'entrée (input_shape)
    # Cette couche est connectée à l'entrée du réseau
    model.add(Dense(n_neuron_per_layer, input_shape=(n_obs,), activation=activation))

    # Ajout des couches cachées supplémentaires, si spécifié
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))

    # Ajout de la couche de sortie avec n_action neurones et activation linéaire
    # (utilisé souvent pour des tâches de régression ou Q-learning)
    model.add(Dense(n_action, activation='linear'))

    # Compilation du modèle avec la fonction de perte et l'optimiseur Adam
    model.compile(loss=loss, optimizer=Adam())

    # Affichage du résumé du modèle (architecture)
    print(model.summary())

    # Retour du modèle prêt à l'emploi
    return model
