# Modifica i file *_channels.tsv sostituendo globalmente "EEG" e "µV" nelle colonne "type" e "units" dove era presente N/A
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

channels_files = list(root.rglob("*_channels.tsv"))

for file in channels_files:
    df = pd.read_csv(file, sep="\t")

    # sostituzione globale coerente
    df["type"] = "EEG"
    df["units"] = "µV"

    df.to_csv(file, sep="\t", index=False)

    print(f"Fixed: {file}")