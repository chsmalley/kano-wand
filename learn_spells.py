import os
import sys
import glob
from typing import Tuple, List
import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
pd.options.plotting.backend = "plotly"
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

Quaternion = Tuple[float, float, float, float]

def make_quat(x: pd.Series):
    return np.array((x["x"], x["y"], x["z"], x["w"]))

def make_euler(x: pd.Series):
    r = R.from_quat(x["q0"])
    euler = r.as_euler('zyx', degrees=True)
    print(f"euler: {euler}")
    return euler

def plot_spell(filename: str):
    title = os.path.splitext(
        os.path.basename(filename))[0]
    df = pd.read_csv(filename, index_col="time")
    # Add distance to dataframe
    df["q0"] = df.apply(func=make_quat, axis=1)
    df["q1"] = df["q0"].shift(1)
    df["euler"] = df.apply(func=make_euler, axis=1)
    # df["dist1"] = df.apply(func=angle_distance, axis=1)
    # df["dist2"] = df.apply(func=euclidean_distance, axis=1)
    print(df)
    # Test dtw
    dist = DTW(df["q0"], df["q1"])

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

#custom metric
def DTW(a: List[float], b: List[float]):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    print(cumdist[an, bn])
    return cumdist[an, bn]

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