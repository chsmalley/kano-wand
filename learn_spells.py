import os
import sys
import glob
from typing import Tuple
import pandas as pd
import numpy as np
import math
pd.options.plotting.backend = "plotly"

Quaternion = Tuple[float, float, float, float]

def make_quat(x: pd.Series):
    return np.array((x["x"], x["y"], x["z"], x["w"]))

def plot_spell(filename: str):
    title = os.path.splitext(
        os.path.basename(filename))[0]
    df = pd.read_csv(filename, index_col="time")
    # Add distance to dataframe
    # df["q0"] = df.apply(func=make_quat, axis=1)
    # df["q1"] = df["q0"].shift(1)
    # df["dist1"] = df.apply(func=angle_distance, axis=1)
    # df["dist2"] = df.apply(func=euclidean_distance, axis=1)
    print(df)
    # df["dist"] = df.apply()
    fig = df.plot(title=filename)
    # fig = df.plot(title=filename).get_figure()
    # fig.savefig("tmp.png")
    fig.show()
    # df.plot(x="time", )

def angle_distance(row: pd.Series):
    """
    Assuming these are unit quaternions used to represent rotations
    """
    if not isinstance(row["q1"], np.ndarray):
        return np.nan
    theta = math.acos(np.dot(row["q0"],
                             row["q1"]))
    if theta > math.pi / 2:
        theta = math.pi - theta
    return theta

def euclidean_distance(row: pd.Series):
    if not isinstance(row["q1"], np.ndarray):
        return np.nan
    else:
        return math.dist(row["q0"], row["q1"])

if __name__ == '__main__':
    dirname = sys.argv[1]
    print(f"dirname: {dirname}")
    for filename in glob.glob(dirname  + "*.csv"):
        print(filename)
        plot_spell(filename)