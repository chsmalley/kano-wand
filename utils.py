from typing import List, Dict
import pandas as pd
import json

def mean_square_error(a: List[float], b: List[float]) -> float:
    # Check number of samples in lists
    diff = len(a) - len(b)
    if diff > 0:
        n = (len(a) + diff // 2) // diff
        a = [v for i, v in enumerate(a) if (i) % n != 0]
    if diff < 0:
        n = (len(b) + abs(diff) //2) // abs(diff)
        b = [v for i, v in enumerate(b) if (i) % n != 0]
    # Now that the lists are resampled compute distance
    print(f"lengths: {len(a)} {len(b)}")
    dist = sum([(a_ - b_)**2 for a_, b_ in zip(a, b)]) / len(a)
    return dist

def df_mean_square_error(df_a: pd.DataFrame, df_b: pd.DataFrame, signals: List[str]) -> float:
    total_error = sum([mean_square_error(df_a[s].values, df_b[s].values)
                       for s in signals])
    return total_error

def classify_spell(performed_spell: pd.DataFrame,
                   spell_data: Dict[str, str]) -> str:
    spell_results: Dict[str, float] = {}
    for spell, spell_file in spell_data.items():
        print(f"{spell}")
        spell_df = pd.read_csv(spell_file, index_col="time")
        spell_results = {}
        spell_results[spell] = \
            df_mean_square_error(spell_df, performed_spell, ["acc_x", "acc_y", "acc_z"])
    print(json.dumps(spell_results, indent=4))
    print(f"you performed: {min(spell_results, key=spell_results.get)}")
    return min(spell_results, key=spell_results.get)