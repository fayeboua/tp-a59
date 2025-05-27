# === Importation des bibliothèques ===
import pickle                         # Pour sauvegarder/charger des objets Python
import matplotlib.pyplot as plt       # Pour tracer les graphiques (portefeuille, prix, etc.)
import time                           # Pour manipuler les timestamps
import numpy as np                    # Pour manipuler les tableaux numériques
import argparse                       # Pour gérer les arguments en ligne de commande
import re                             # Pour extraire les timestamps depuis les noms de fichiers

# === Importation des modules du projet ===
from envs import TradingEnv           # Environnement de trading personnalisé (type gym.Env)
from agent import DQNAgent            # Agent DQN défini dans agent.py
from utils import get_data, get_data_yf, get_scaler, maybe_make_dir  # Fonctions utilitaires



def main():
    # === Parsing des arguments en ligne de commande ===
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=2000, help='Number of episodes to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000, help='Initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True, help='"train" or "test"')
    parser.add_argument('-w', '--weights', type=str, help='Trained model weights to load for test')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum steps per episode')
    args = parser.parse_args()

    # Création des répertoires de sauvegarde
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    # Création d’un identifiant temporel
    timestamp = time.strftime('%Y%m%d%H%M')

    # Chargement des données (ici depuis Yahoo Finance)
    data = np.around(get_data_yf())
    data_size = data.shape[1]

    # Séparation en train/test
    train_data_size = int(data_size * 0.8)
    train_data = data[:, :train_data_size]  # 80% des lignes
    test_data = data[:, train_data_size:]   # 20% restantes

    print('train_data shape:', train_data.shape)

    # Création de l’environnement et de l’agent
    env = TradingEnv(train_data if args.mode == 'train' else test_data,
                     args.initial_invest,
                     max_steps=args.max_steps)

    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    # Chargement des poids (si en mode test)
    if args.mode == 'test':
        if not args.weights:
            raise ValueError("Please specify weights file with --weights for test mode")
        agent.load(args.weights)
        timestamp = re.findall(r'\d{12}', args.weights)[0]

    # Création d’un scaler pour normaliser l’observation
    scaler = get_scaler(env)
    portfolio_value = []

    # Boucle principale sur les épisodes
    for e in range(args.episode):
        state = env.reset()
        state = scaler.transform([state])

        # Préparation à la visualisation du dernier épisode (test)
        # === Réinitialiser pour le dernier épisode (test) ===
        if args.mode == 'test' and e == args.episode - 1:
            value_over_time = []
            price_history = []
            buy_points = []
            sell_points = []

        # Boucle sur les pas de temps
        for t in range(env.max_steps or env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])

            # Entraînement ou Test
            # Entraînement : on stocke les transitions dans la mémoire
            # Test : on historise les valeurs du portefeuille et les points d’achat/vente
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            else:
                if e == args.episode - 1:
                    value_over_time.append(info['cur_val'])
                    price_history.append(info['price'][0])
                    action_vec = info.get("action_vec", [])
                    if action_vec:
                        if action_vec[0] == 0:
                            sell_points.append((t, info['cur_val']))
                        elif action_vec[0] == 2:
                            buy_points.append((t, info['cur_val']))

            # Mise à jour de l'état pour le prochain pas de temps
            state = next_state

            # Fin d’épisode
            if done:
                print(f"Episode {e + 1}/{args.episode}, Final Portfolio Value: {info['cur_val']}")
                portfolio_value.append(info['cur_val'])
                break

            # Apprentissage par replay
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)

        # Sauvegarde périodique des poids (tous les 10 épisodes)
        if args.mode == 'train' and (e + 1) % 10 == 0:
            agent.save(f'weights/{timestamp}-dqn.weights.h5')

    # === Sauvegarde des valeurs du portefeuille ===
    with open(f'portfolio_val/{timestamp}-{args.mode}.p', 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    # === Visualisation finale (dernier épisode uniquement, en test) ===
    if args.mode == 'test':
        buy_x, buy_y = zip(*buy_points) if buy_points else ([], [])
        sell_x, sell_y = zip(*sell_points) if sell_points else ([], [])
        val_x = list(range(len(value_over_time)))
        val_y = value_over_time

        # Courbe de la valeur du portefeuille
        plt.figure(figsize=(12, 6))
        plt.plot(val_x, val_y, label='Portfolio Value', color='blue')
        plt.scatter(buy_x, buy_y, color='green', marker='o', label='Buy', s=20)
        plt.scatter(sell_x, sell_y, color='red', marker='o', label='Sell', s=20)
        plt.title("Trading Agent Performance (Last Episode)")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'portfolio_val/{timestamp}_trading_plot.png')
        plt.show()

        # Courbe du prix de l'action avec les points d'achat/vente
        plt.figure(figsize=(12, 6))
        price_x = list(range(len(price_history)))
        plt.plot(price_x, price_history, label='Stock Price', color='orange')
        plt.scatter(buy_x, [price_history[i] for i in buy_x], color='green', label='Buy', s=20)
        plt.scatter(sell_x, [price_history[i] for i in sell_x], color='red', label='Sell', s=20)
        plt.title("Buy/Sell Points on Stock Price (Last Episode)")
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'portfolio_val/{timestamp}_price_plot.png')
        plt.show()


if __name__ == '__main__':
    main()
