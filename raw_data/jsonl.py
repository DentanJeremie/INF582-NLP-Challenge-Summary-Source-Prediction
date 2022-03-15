import pandas as pd

df = pd.read_json('test_set.json')

df.to_json("test_set.jsonl", orient="records", lines=True)