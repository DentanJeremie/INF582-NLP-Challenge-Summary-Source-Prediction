# INF582-NLP_challenge
Readme associated to the git repository : https://github.com/paultheron-X/INF582-NLP_challenge.git
Authors : Jérémie Dentan, Louis Gautier, Paul Théron

## Getting started
In order to set up the project, please use a virtual environment that can be created and activated with
```bash
python3 -m venv .venv
source ./.venv/bin/activate
```
Then, install the required libraries with
```bash
pip install --upgrade pip
pip3 install -r requirements.txt
```

## Dowload data and compute features
To reproduce the results of the paper, please run :
```
bash sh/data_download.sh
bash preprocessing.sh
```

The preprocessing time is about 1h. However, we provide the computed `.csv` files in `processed_data` directory.

## Prediction
Simply run:
```
python main.py
```
or, for the same algorithm with fine-tuned parameters (same performance yet) :
```
python main_xgbtuned.py
```