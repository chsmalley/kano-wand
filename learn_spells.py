"""
How to run
cd into repo
$ python learn_spells.py test_data/spells1/ test_data/spells3/
Note: spells1 and spells3 were captured by the same person. 
spells2 was captured by a different person on the first time looking at the 
spell chart. Some mistakes may have been made following the spell chart.
"""

import os
import sys
import glob
import json
from typing import Tuple, List, Dict, Callable
import pandas as pd
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
# pd.options.plotting.backend = "plotly"
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from scipy.stats import mode
from pyquaternion import Quaternion
import plotly.express as px
import plotly.graph_objects as go

# Quaternion = Tuple[float, float, float, float]

LABELS = {
    1: 'AGUAMENTI',
    2: 'AVIS',
    3: 'ENGORGIO',
    4: 'EXPELLIARMUS',
    5: 'FLIPENDO',
    6: 'INCENDIO',
    7: 'LOCOMOTOR',
    8: 'LUMOS',
    9: 'REDUCIO',
    10: 'REDUCTO',
    11: 'STUPEFY',
    12: 'WINGARDIUM_LEVIOSA'
}


class Acceleration:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def make_quat(x: pd.Series):
    # print(f"in: {x}")
    # a = np.array([x["x"], x["y"], x["z"], x["w"]])
    # return a / np.linalg.norm(a)
    # print(f"out: {Quaternion(a).unit}")
    return Quaternion(x["w"], x["x"], x["y"], x["z"]).unit

def angle_distance(x, y):
    """
    Assuming these are unit quaternions used to represent rotations
    """
    theta = math.acos(np.dot(x, y))
    if theta > math.pi / 2:
        theta = math.pi - theta
    return theta

def accel_distance(a, b):
    # return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)
    return (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2

def dtw_distance(
    ts_a,
    ts_b,
    d = accel_distance,
    # d = lambda x, y: abs(x - y),
    # d = lambda x, y: math.dist(x, y),
    # d: Callable = lambda x, y: Quaternion.distance(x, y),
    max_warping_window: int=10,
) -> float:
    """Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.

    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared
    
    d : DistanceMetric object (default = math.dist(x, y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function
    
    Returns
    -------
    DTW distance between A and B
    """

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * np.ones((M, N))

    # Initialize the first row and column
    # print(f"dist {ts_a[0]}, {ts_b[0]}")
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                        min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]

def distance_matrix(x, y, subsample_step=1):
    """Computes the M x N distance matrix between the training
    dataset (x) and testing dataset (y) using the DTW distance measure
    
    Arguments
    ---------
    x : array of shape [n_samples, n_timepoints]
    y : array of shape [n_samples, n_timepoints]
    
    Returns
    -------
    Distance matrix between each item of x and y with
        shape [training_n_samples, testing_n_samples]
    """
    
    # Compute the distance matrix        
    dm_count = 0
    # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
    # when x and y are the same array
    if(np.array_equal(x, y)):
        x_s = np.shape(x)
        dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
        
        for i in range(0, x_s[0] - 1):
            for j in range(i + 1, x_s[0]):
                dm[dm_count] = dtw_distance(x[i, ::subsample_step],
                                            y[j, ::subsample_step])
                dm_count += 1
        
        # Convert to squareform
        dm = distance.squareform(dm)
        return dm
    
    # Compute full distance matrix of dtw distnces between x and y
    else:
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0])) 
        dm_size = x_s[0]*y_s[0]
    
        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                dm[i, j] = dtw_distance(x[i, ::subsample_step],
                                        y[j, ::subsample_step])
                dm_count += 1
        return dm

def predict(
    dist_matrix,
    x,
    labels: List[str],
    n_neighbors=1
    ):
    """Predict the class labels or probability estimates for 
    the provided data

    Arguments
    ---------
      
      x : array of shape [n_samples, n_timepoints]
          Array containing the testing data set to be classified
      
      
    Returns
    -------
      2 arrays representing:
          (1) the predicted class labels 
          (2) the knn label count probability
    """

    # Identify the k nearest neighbors
    knn_idx = dist_matrix.argsort()[:, :n_neighbors]

    # Identify k nearest labels
    print(f"knn_idx: {knn_idx}")
    knn_labels = labels[knn_idx]
    
    # Model Label
    mode_data = mode(knn_labels, axis=1)
    mode_label = mode_data[0]
    mode_proba = mode_data[1] / n_neighbors

    return mode_label.ravel(), mode_proba.ravel()


def make_euler(x: pd.Series):
    r = R.from_quat(x["q0"])
    euler = r.as_euler('zyx', degrees=True)
    return euler

def learn_spell(train_spells: Dict[str, str], test_spells: Dict[str, str]):
    spell_results = {}
    for train_spell, train_spell_file in train_spells.items():
        train_df = pd.read_csv(train_spell_file, index_col="time")
        train_df["time_from_start"] = [t - list(train_df.index)[0]
                                       for t in list(train_df.index)]
        acc_x = train_df["acc_x"].to_numpy()
        acc_y = train_df["acc_y"].to_numpy()
        acc_z = train_df["acc_z"].to_numpy()
        train_array = np.array([Acceleration(x, y, z) for x, y, z in zip(acc_x, acc_y, acc_z)])
        train_array = train_array.reshape((1, train_array.shape[0]))
        spell_results[train_spell] = {}
        for test_spell, test_spell_file in test_spells.items():
            test_df = pd.read_csv(test_spell_file, index_col="time")
            test_df["time_from_start"] = [t - list(test_df.index)[0]
                                           for t in list(test_df.index)]
            # Add distance to dataframe
            acc_x = test_df["acc_x"].to_numpy()
            acc_y = test_df["acc_y"].to_numpy()
            acc_z = test_df["acc_z"].to_numpy()
            test_array = np.array([Acceleration(x, y, z) for x, y, z in zip(acc_x, acc_y, acc_z)])
            test_array = test_array.reshape((1, test_array.shape[0]))
            dist_matrix = distance_matrix(train_array, test_array)
            spell_results[train_spell][test_spell] = dist_matrix[0][0]
            # a, b = predict(dist_matrix, test_array, np.array([1]))
            # print(f"predict results: {a} {b}")
            # fig = px.line(train_df, x="time_from_start", y=["acc_x", "acc_y", "acc_z"], title=train_spell)
            # fig.add_trace(go.Scatter(x=test_df["time_from_start"],
            #                          y=test_df["acc_x"], name=f"acc_x {test_spell}"))
            # fig.add_trace(go.Scatter(x=test_df["time_from_start"],
            #                          y=test_df["acc_y"], name=f"acc_y {test_spell}"))
            # fig.add_trace(go.Scatter(x=test_df["time_from_start"],
            #                          y=test_df["acc_z"], name=f"acc_z {test_spell}"))
            # fig.show()
    print(json.dumps(spell_results, indent=4))
    for spell in spell_results.keys():
        print(f"{spell}: {min(spell_results[spell], key=spell_results[spell].get)}")
    # Test dtw
    # dist = DTW(train_df["q0"], test_df["q0"])
    # dist = DTW(np.array([1] * 100), np.array([1] * 100))
    # print(f"dtw dist: {dist}")
    # Plot some stuff
    # df["dist"] = df.apply()
    # fig = df.plot(title=filename)
    # fig = df.plot(title=filename).get_figure()
    # fig.savefig("tmp.png")
    # fig.show()
    # df.plot(x="time", )


def learn_spell_quaternion(train_spells: Dict[str, str], test_spells: Dict[str, str]):
    spell_results = {}
    for train_spell, train_spell_file in train_spells.items():
        train_df = pd.read_csv(train_spell_file, index_col="time")
        train_df["q0"] = train_df.apply(func=make_quat, axis=1)
        train_array = train_df["q0"].to_numpy()
        # Calculate distance of every point to center point
        train_center = train_array[len(train_array) // 2]
        train_array = np.array([Quaternion.distance(x, train_center)
                                for x in train_array])
        train_array = train_array.reshape((1, train_array.shape[0]))
        spell_results[train_spell] = {}
        for test_spell, test_spell_file in test_spells.items():
            test_df = pd.read_csv(test_spell_file, index_col="time")
            # Add distance to dataframe
            test_df["q0"] = test_df.apply(func=make_quat, axis=1)
            test_array = test_df["q0"].to_numpy()
            # Calculate distance of every point to center point
            test_center = test_array[len(test_array) // 2]
            test_array = np.array([Quaternion.distance(x, test_center)
                                   for x in test_array])
            test_array = test_array.reshape((1, test_array.shape[0]))
            dist_matrix = distance_matrix(train_array, test_array)
            spell_results[train_spell][test_spell] = dist_matrix[0][0]
            # a, b = predict(dist_matrix, test_array, np.array([1]))
            # print(f"predict results: {a} {b}")
    print(json.dumps(spell_results, indent=4))
    # Test dtw
    # dist = DTW(train_df["q0"], test_df["q0"])
    # dist = DTW(np.array([1] * 100), np.array([1] * 100))
    # print(f"dtw dist: {dist}")
    # df["dist"] = df.apply()
    # fig = df.plot(title=filename)
    # fig = df.plot(title=filename).get_figure()
    # fig.savefig("tmp.png")
    # fig.show()
    # df.plot(x="time", )

#custom metric
def DTW(a: List[float], b: List[float]):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1), b.reshape(-1,1))
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
    train_dirname = sys.argv[1]
    test_dirname = sys.argv[2]
    train_spells = {}
    for filename in glob.glob(train_dirname + "*.csv"):
        name = os.path.splitext(os.path.basename(filename))[0]
        train_spells[name] = filename
    test_spells = {}
    for filename in glob.glob(test_dirname + "*.csv"):
        name = os.path.splitext(os.path.basename(filename))[0]
        test_spells[name] = filename
    learn_spell(test_spells, train_spells)