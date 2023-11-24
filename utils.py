from typing import List, Dict, Tuple
import pandas as pd
import json
import os

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
    dist = sum([(a_ - b_)**2 for a_, b_ in zip(a, b)]) / len(a)
    return dist

def df_mean_square_error(df_a: pd.DataFrame, df_b: pd.DataFrame, signals: List[str]) -> float:
    total_error = sum([mean_square_error(df_a[s].values, df_b[s].values)
                       for s in signals])
    return total_error

def get_nearest_neighbors(
    train_data: Dict[str, pd.DataFrame],
    test_data: pd.DataFrame,
    k: int
) -> List[Tuple[str, float]]:
    # Calculate distance between test data and all training data
    distances = []
    for key, val in train_data.items():
        spell_name = os.path.splitext(key)[0]
        distances.append((spell_name, df_mean_square_error(val, test_data, ["acc_x", "acc_y", "acc_z"])))
    # Sort distances
    distances.sort(key = lambda x: x[1])
    # Return closest k neighbors
    return distances[:k]

def classify_spell(
    performed_spell: pd.DataFrame,
    spell_data: Dict[str, pd.DataFrame]
) -> str:
    neighbors = get_nearest_neighbors(spell_data, performed_spell, 3)
    # Get most common neighbor
    neighbor_names = [n[0] for n in neighbors]
    return max(set(neighbor_names), key=neighbor_names.count)