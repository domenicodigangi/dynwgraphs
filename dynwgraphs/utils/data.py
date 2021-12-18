#########
#Filename: d:\pCloud\Dynamic_Networks\repos\DynWGraphsPaper\src\dynwgraphs\dynwgraphs\utils\data
#Path: d:\pCloud\Dynamic_Networks\repos\DynWGraphsPaper\src\dynwgraphs\dynwgraphs\utils
#Created Date: Thursday, October 21st 2021, 3:37:57 pm
#Author: Domenico Di Gangi
#
#Copyright (c) 2021 Your Company
########


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



def cluster_2_and_fill_less_freq(obs_in):

    obs = obs_in.copy()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(obs)

    inds = kmeans.labels_ == kmeans.cluster_centers_.argmax()

    obs[~inds] = np.nan

    # if inds.mean() > 0:
    #     obs[~inds] = np.nan
    # else:
    #     obs[inds] = np.nan



    obs = pd.Series(obs.flatten()).fillna(method="ffill").values

    nainds = np.isnan(obs)
    if np.any(nainds):
        obs[np.where(nainds)] = obs[np.where(~nainds)].mean()
    return obs
