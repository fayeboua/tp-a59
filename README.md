# TP-A59

## Aperçu

Voici le code associé à [cette vidéo](https://youtu.be/rRssY6FrTvU) sur Youtube par Siraj Raval concernant le Q Learning pour le trading, dans le cadre du cours Move 37 à la [School of AI](https://www.theschool.ai). Les crédits pour ce code reviennent à [ShuaiW](https://github.com/ShuaiW/teach-machine-to-trade).

Article associé : [Teach Machine to Trade](https://shuaiw.github.io/2018/02/11/teach-machine-to-trade.html)

### Dépendances

Ce projet prend désormais en charge **Python 3.12.3**.

Pour configurer votre environnement et installer toutes les bibliothèques requises :

1. **(Recommandé)** Créez un environnement virtuel :

```bash
python3.12 -m venv myenv
# Sur macOS/Linux :
source myenv/bin/activate
# Sur Windows :
myenv\Scripts\activate
```

2. **Mettez à jour pip (optionnel mais recommandé) :**

    ```bash
    python -m pip install --upgrade pip
    ```

3. **Installez les dépendances :**

    ```bash
    pip install -r requirements.txt
    ```

> **Remarque :** Si vous rencontrez des problèmes avec certains paquets, assurez-vous que toutes les dépendances dans `requirements.txt` sont compatibles avec Python 3.12.3. Vous devrez peut-être mettre à jour certaines versions de paquets.

### Table des matières

* `agent.py` : un agent Deep Q learning
* `envs.py` : un environnement de trading simple avec 3 actions
* `model.py` : un perceptron multicouche comme estimateur de fonction
* `utils.py` : quelques fonctions utilitaires
* `run.py` : logique d'entraînement/test
* `requirements.txt` : toutes les dépendances
* `data/` : 3 fichiers csv avec les prix des actions IBM, MSFT et QCOM du 3 janvier 2000 au 27 décembre 2017 (5629 jours). Les données ont été récupérées via [Alpha Vantage API](https://www.alphavantage.co/)
<https://github.com/llSourcell/Q-Learning-for-Trading>

### Comment exécuter

**Pour entraîner un agent Deep Q**, exécutez :

```bash
python run.py --mode train
```

Il existe d'autres paramètres ; consultez le script `run.py` pour plus de détails. Après l'entraînement, un modèle entraîné et l'historique de la valeur du portefeuille à la fin de chaque épisode seront sauvegardés sur le disque.

Exemple :

```bash
python run.py --mode train -e 100 -i 10000 --max_steps 100
```

**Pour tester les performances du modèle**, exécutez :

```bash
python run.py --mode test --weights <trained_model>
```

Remplacez `<trained_model>` par le chemin vers votre fichier de poids du modèle entraîné (par exemple, `weights/202505182127-dqn.weights.h5`). Cela évaluera les performances de l'agent sur le jeu de test et enregistrera l'historique de la valeur du portefeuille pour chaque épisode.

Vous pouvez spécifier des paramètres supplémentaires si besoin. Voici les principales options configurables disponibles dans `run.py` :

* `-e`, `--episode` : Nombre d'épisodes à exécuter (défaut : 2000)
* `-b`, `--batch_size` : Taille du batch pour le replay mémoire (défaut : 32)
* `-i`, `--initial_invest` : Montant de l'investissement initial (défaut : 20000)
* `-m`, `--mode` : "train" ou "test" (obligatoire)
* `-w`, `--weights` : Poids du modèle entraîné à charger pour le test
* `--max_steps` : Nombre maximal d'étapes par épisode (défaut : None)

**Exemple d'utilisation :**

```bash
python run.py --mode test --weights weights/202505182127-dqn.weights.h5 --max_steps 500 --episode 50 --batch_size 64 --initial_invest 50000
```

Pour la liste complète des options configurables et leurs descriptions, consultez les détails dans `run.py`.
