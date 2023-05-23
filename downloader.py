import os
import urllib
import urllib.request

TRAIN_DATA_URL = "https://storage.googleapis.com/deepmind-gutenberg/train/{}.txt"
VALIDATION_DATA_URL = (
    "https://storage.googleapis.com/deepmind-gutenberg/validation/{}.txt"
)

PLATO_TEXT_IDS = [
    1580,
    1657,
    55201,
    1635,
    1658,
    1744,
    1672,
    1643,
    1687,
    1584,
    1656,
    1497,
    1616,
    1598,
]

output_dir = "data/plato"
os.makedirs(output_dir, exist_ok=True)

for id in PLATO_TEXT_IDS:
    try:
        urllib.request.urlretrieve(
            TRAIN_DATA_URL.format(id), os.path.join(output_dir, f"{id}.txt")
        )
    except urllib.error.HTTPError:
        urllib.request.urlretrieve(
            VALIDATION_DATA_URL.format(id), os.path.join(output_dir, f"{id}.txt")
        )
