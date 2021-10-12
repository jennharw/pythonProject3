#ranking

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pandas.plotting import table
import subprocess
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import distance

from tqdm.auto import tqdm
import pickle
import gc
import json
import pandas as pd

# == recnn ==
import sys
import os
sys.path.append(os.getcwd())

import recnn2

from recnn2.data.utils import get_base_batch

#cuda = torch.device('cuda')
frame_size = 10
meta = json.load(open('data/parsed/omdb.json'))
tqdm.pandas()

frame_size = 10
batch_size = 1
dirs = recnn2.data.env.DataPath(
    base=os.path.join(os.getcwd(), 'data/'),
    embeddings="embeddings/ml20_pca128.pkl",
    ratings="ml-20m/ratings.csv",
    cache="cache/frame_env.pkl", # cache will generate after you run
    use_cache=True
)
env = recnn2.data.env.FrameEnv(dirs, frame_size, batch_size)

ddpg = recnn2.nn.models.Actor(1290, 128, 256)#.to(cuda)
td3 = recnn2.nn.models.Actor(1290, 128, 256)#.to(cuda)
ddpg.load_state_dict(torch.load('models/ddpg_policy.pt'))
#td3.load_state_dict(torch.load('../../models/td3_policy.pt'))



test_batch = next(iter(env.test_dataloader))
state, action, reward, next_state, done = get_base_batch(test_batch)

def rank(gen_action, metric):
    scores = []
    for i in env.base.key_to_id.keys():
        if i == 0 or i == '0':
            continue
        scores.append([i, metric(env.base.embeddings[env.base.key_to_id[i]], gen_action)])
    scores = list(sorted(scores, key = lambda x: x[1]))
    scores = scores[:10]
    ids = [i[0] for i in scores]
    for i in range(10):
        scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
                                'Genre', 'Language', 'Released', 'imdbRating']])
        # scores[i][3] = ' '.join([genres_dict[i] for i in scores[i][3]])

    indexes = ['id', 'score', 'Title', 'Genre', 'Language', 'Released', 'imdbRating']
    table_dict = dict([(key,[i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    table = pd.DataFrame(table_dict)
    return table

#DDPG
def rank_result():
    print(test_batch)
    ddpg_action = ddpg(state)
    # pick random action
    ddpg_action = ddpg_action[np.random.randint(0, state.size(0), 1)[0]].detach().cpu().numpy()

    return rank(ddpg_action, distance.euclidean)
