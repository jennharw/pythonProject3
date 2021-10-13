import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import pickle
import json
import copy
import pandas as pd
import random
from tqdm.auto import tqdm

import torch
from scipy.spatial import distance

import recnn
from recnn.nn.models import Actor
from recnn.data.utils import make_items_tensor

import sys
import os
sys.path.append(os.getcwd())

ML20MPATH = 'data/ml20m/'
MODELSPATH = 'data/models/'
DATAPATH = 'data/'
SHOW_TOPN_MOVIES = 200

ML20MPATH = 'thesis/embedding/'
MODELSPATH = 'thesis/models/'
DATAPATH = 'thesis/parsed'
SHOW_TOPN_MOVIES = 200


def rank(gen_action, metric, k):
    scores = []
    movie_embeddings_key_dict = load_mekd()
    meta = load_omdb_meta()

    for i in movie_embeddings_key_dict.keys():
        if i == 0 or i == '0':
            continue
        scores.append([i, metric(movie_embeddings_key_dict[i], gen_action)])
    scores = list(sorted(scores, key = lambda x: x[1]))
    scores = scores[:k]
    ids = [i[0] for i in scores]

    print(ids)
    #thesis = pd.read_excel('/data/workspace/holly0015/test_project1/project1/src/thesis/parsed/thesis_list.xlsx')

    # for i in range(k):
    #     scores[i].extend([thesis['article_title'][ids]])
    #
    #
    # for i in range(k):
    #     scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
    #                                                                         'Genre', 'imdbRating']])
    # indexes = ['id', 'score', 'Title', 'Genre', 'imdbRating']
    # table_dict = dict([(key, [i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    #table = pd.DataFrame(table_dict)
    table = pd.DataFrame(scores)
    return table



@st.cache
def get_mov_base():
    links = load_links()
    #movies_embeddings_tensor, key_to_id, id_to_key = get_embeddings()
    meta = load_omdb_meta()

    popular = pd.read_excel('/data/workspace/holly0015/test_project1/project1/src/thesis/parsed/thesis_list.xlsx')[:SHOW_TOPN_MOVIES]
    mov_base = {}
    pop_list = list(popular['movieId'])

    for i in range(len(popular)):
        movie_id = popular['movieId'][i]
        movie_title = popular['article_title'][i]
        mov_base[int(movie_id)] = movie_title
    # for i, k in list(meta.items()):
    #     print(meta.items)
    #     tmdid = int(meta[i]['tmdbId'])
    #     if tmdid in pop_list:
    #         movieid = pd.to_numeric(links.loc[tmdid]['movieId'])
    #         if isinstance(movieid, pd.Series):
    #             continue
    #         mov_base[int(movieid)] = meta[i]['omdb']['Title']

    return mov_base

@st.cache
def load_mekd():
    file = open('/data/workspace/holly0015/test_project1/project1/src/thesis/embedding/thesis_pca128.pkl', 'rb')
    data = pickle.load(file)
    file.close()
    return data


@st.cache
def load_omdb_meta():
    #return json.load(open('/data/workspace/holly0015/test_project1/project1/src/data/parsed/omdb.json'))
    return pd.read_excel('/data/workspace/holly0015/test_project1/project1/src/thesis/parsed/thesis_list.xlsx')

@st.cache
def load_links():
    #https://raw.githubusercontent.com/awarebayes/recnn-demo/master/app/ml20m/links.csv
    url = 'https://raw.githubusercontent.com/awarebayes/recnn-demo/master/app/ml20m/links.csv'
    #return pd.read_csv('/data/workspace/holly0015/test_project1/project1/src/data/ml-20m/links.csv', index_col='tmdbId')
    return pd.read_csv('/data/workspace/holly0015/test_project1/project1/src/thesis/parsed/links.csv', index_col='thesis_DOI')

def get_embeddings():
    movie_embeddings_key_dict = load_mekd()
    movies_embeddings_tensor, key_to_id, id_to_key = make_items_tensor(movie_embeddings_key_dict)
    return  movies_embeddings_tensor, key_to_id, id_to_key


def load_models():
    ddpg =Actor(1290, 128, 256)#.to(device)
    #td3 = recnn.nn.models.Actor(1290, 128, 256)#.to(device)

    ddpg.load_state_dict(torch.load('/data/workspace/holly0015/test_project1/project1/src/thesis/models/ddpg_policy.pt'))#, map_location=device))
    #td3.load_state_dict(torch.load(MODELSPATH + 'td3_policy.model', map_location=device))
    return {'ddpg': ddpg} #, 'td3': td3}

device = torch.device('cuda') #cpu

def main():
    st.sidebar.header('📰 recnn by @awarebayes 👨‍🔧')

    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["🚀 Get Started", "📽 ️Recommend me a movie", "🔨 Test Recommendation"])
                                    # ,"⛏️ Test Diversity", "🤖 Reinforce Top K"])

    if page == "🚀 Get Started":
        st.subheader('DDPG')
        #render_header()

    if page == "🔨 Test Recommendation":
        st.header("Test the Recommendations")
        models = load_models()

    if page == "📽 ️Recommend me a movie":
        st.header("📽 ️Recommend me a movie")
        mov_base = get_mov_base()
        mov_base_by_title = {v: k for k, v in mov_base.items()}
        movies_chosen = st.multiselect('Choose 10 movies', list(mov_base.values()))
        st.markdown('**{} chosen {} to go**'.format(len(movies_chosen), 10 - len(movies_chosen)))

        if len(movies_chosen) > 10:
            st.error('Please select exactly 10 movies, you have selected {}'.format(len(movies_chosen)))
        if len(movies_chosen) == 10:
            st.success("You have selected 10 movies. Now let's rate them")
        else:
            st.info('Please select 10 movies in the input above')

        if len(movies_chosen) == 10:
            st.markdown('### Rate each movie from 1 to 10')
            ratings = dict([(i, st.number_input(i, min_value=1, max_value=10, value=5)) for i in movies_chosen])
            # st.write('for debug your ratings are:', ratings)

            ids = [mov_base_by_title[i] for i in movies_chosen]
            # st.write('Movie indexes', list(ids))
            embs = load_mekd()
            state = torch.cat([torch.cat([embs[i] for i in ids]), torch.tensor(list(ratings.values())).float() - 5])
            st.write('your state', state)
            state = state.squeeze(0)

            models = load_models()

            algorithm = 'ddpg'
            metric = 'cosine'
            dist = distance.cosine
            topk = st.slider("TOP K items to recommend:", min_value=1, max_value=30, value=7)

            action = models[algorithm].forward(state)
            st.subheader('The neural network thinks you should watch:')

            st.write(rank(action[0].detach().cpu().numpy(), dist, topk))


if __name__ == "__main__":
    main()