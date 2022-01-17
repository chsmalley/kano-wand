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

def dtw_distance(
    ts_a,
    ts_b,
    d = lambda x,y: abs(x-y)
    max_warping_window = 10000,
): -> float
    """Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.

    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared
    
    d : DistanceMetric object (default = abs(x-y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function
    
    Returns
    -------
    DTW distance between A and B
    """

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxint * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in xrange(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in xrange(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(max(1, i - max_warping_window),
                        min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window 
    return cost[-1, -1]

def dist_matrix(x, y, subsample_step=1):
    """Computes the M x N distance matrix between the training
    dataset and testing dataset (y) using the DTW distance measure
    
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
        
        for i in xrange(0, x_s[0] - 1):
            for j in xrange(i + 1, x_s[0]):
                dm[dm_count] = dtw_distance(x[i, ::subsample_step],
                                            y[j, ::subsample_step])
                dm_count += 1
        
        # Convert to squareform
        dm = squareform(dm)
        return dm
    
    # Compute full distance matrix of dtw distnces between x and y
    else:
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0])) 
        dm_size = x_s[0]*y_s[0]
    
        for i in xrange(0, x_s[0]):
            for j in xrange(0, y_s[0]):
                dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                              y[j, ::self.subsample_step])
                # Update progress bar
                dm_count += 1
        return dm

def predict(self, x):
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
    
    dm = self._dist_matrix(x, self.x)

    # Identify the k nearest neighbors
    knn_idx = dm.argsort()[:, :self.n_neighbors]

    # Identify k nearest labels
    knn_labels = self.l[knn_idx]
    
    # Model Label
    mode_data = mode(knn_labels, axis=1)
    mode_label = mode_data[0]
    mode_proba = mode_data[1]/self.n_neighbors

    return mode_label.ravel(), mode_proba.ravel()


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