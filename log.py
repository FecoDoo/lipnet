from pathlib import Path

log_path = Path(__file__).resolve().parent.joinpath("logs")

files = log_path.glob("*.log")

res = {}
for f in files:

    res[f.stem] = []
    with open(f, "r") as fp:
        text = fp.read()

    res[f.stem] = [text[i : i + 6] for i in range(0, int(len(text) / 6))]

import json

with open(log_path.joinpath("error.json"), "w") as fp:
    json.dump(res, fp)
