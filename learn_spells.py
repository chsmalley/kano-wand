import os
import sys
import glob
from typing import Tuple
import pandas as pd
import numpy as np
import math

Quaternion = Tuple(float, float, float, float)

def plot_spell(filename: str):
    title = os.path.splitext(
        os.path.basename(filename))[0]
    df = pd.read_csv(filename, index="time")
    df.plot(title=filename)
    # df.plot(x="time", )

def angle_distance(q0: Quaternion, q1: Quaternion):
    """
    Assuming these are unit quaternions used to represent rotations
    """
    theta = math.acos(np.dot(np.array(q0), np.array(q1)))
    if theta > math.pi / 2:
        theta = math.pi - theta
    return theta

def euclidean_distance(q0: Quaternion, q1: Quaternion):
    return math.dist(q0, q1)

if __name__ == '__main__':
    dirname = sys.argv[1]
    for filename in glob.glob("*.csv"):
        print(filename)