# TP-A59

## Overview

This is the code for [this](https://youtu.be/rRssY6FrTvU) video on Youtube by Siraj Raval on Q Learning for Trading as part of the Move 37 course at [School of AI](https://www.theschool.ai). Credits for this code go to [ShuaiW](https://github.com/ShuaiW/teach-machine-to-trade).

Related post: [Teach Machine to Trade](https://shuaiw.github.io/2018/02/11/teach-machine-to-trade.html)

### Dependencies

Python 2.7. To install all the libraries, run `pip install -r requirements.txt`

### Table of content

* `agent.py`: a Deep Q learning agent
* `envs.py`: a simple 3-stock trading environment
* `model.py`: a multi-layer perceptron as the function approximator
* `utils.py`: some utility functions
* `run.py`: train/test logic
* `requirement.txt`: all dependencies
* `data/`: 3 csv files with IBM, MSFT, and QCOM stock prices from Jan 3rd, 2000 to Dec 27, 2017 (5629 days). The data was retrieved using [Alpha Vantage API](https://www.alphavantage.co/)
<https://github.com/llSourcell/Q-Learning-for-Trading>

### How to run

**To train a Deep Q agent**, run `python run.py --mode train`. There are other parameters and I encourage you look at the `run.py` script. After training, a trained model as well as the portfolio value history at episode end would be saved to disk.
example: python run.py --mode train -e 100 -i 10000 --max_steps 100

**To test the model performance**, run:

```bash
python run.py --mode test --weights <trained_model>
```

Replace `<trained_model>` with the path to your trained model weights file (e.g., `weights/202505182127-dqn.weights.h5`). This will evaluate the agent's performance on the test dataset and save the portfolio value history for each episode.

You can specify additional parameters as needed. Here are the main configurable options available in `run.py`:

* `-e`, `--episode`: Number of episodes to run (default: 2000)

* `-b`, `--batch_size`: Batch size for memory replay (default: 32)

* `-i`, `--initial_invest`: Initial investment amount (default: 20000)

* `-m`, `--mode`: "train" or "test" (required)

* `-w`, `--weights`: Trained model weights to load for test
  
* `--max_steps`: Maximum steps per episode (default: None)

**Example usage:**

```bash
python run.py --mode test --weights weights/202505182127-dqn.weights.h5 --max_steps 500 --episode 50 --batch_size 64 --initial_invest 50000
```

For a full list of configurable options and their descriptions, see the details in `run.py`.
