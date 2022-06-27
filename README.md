# INF582-NLP_challenge
Paper associated to this git repository : https://www.researchgate.net/publication/361569713_INF582_NLP_Challenge_Summary_Source_Prediction <br>
Readme associated to the git repository : https://github.com/paultheron-X/INF582-NLP_challenge.git <br>
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

## Documentation
An academic description of our work is available :
- Report : [here](/documentation/INF582_Report.pdf)
- Presentation : [here](/documentation/INF582_slides.pdf)

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