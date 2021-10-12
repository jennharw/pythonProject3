import pickle
import json
import pandas as pd
from recnn2.data.utils import make_items_tensor

def load_mekd():
    return pickle.load(open('/data/workspace/holly0015/test_project1/project1/src/data/embeddings/ml20_pca128.pkl', 'rb'))
def get_embeddings():
    movie_embeddings_key_dict = load_mekd()
    movies_embeddings_tensor, key_to_id, id_to_key = make_items_tensor(movie_embeddings_key_dict)
    return  movies_embeddings_tensor, key_to_id, id_to_key

def load_omdb_meta():
    return json.load(open('/data/workspace/holly0015/test_project1/project1/src/data/parsed/omdb.json'))

def load_links():
    #https://raw.githubusercontent.com/awarebayes/recnn-demo/master/app/ml20m/links.csv
    url = 'https://raw.githubusercontent.com/awarebayes/recnn-demo/master/app/ml20m/links.csv'
    return pd.read_csv('/data/workspace/holly0015/test_project1/project1/src/data/ml-20m/links.csv', index_col='tmdbId')

# links = load_links()
# movies_embeddings_tensor, key_to_id, id_to_key = get_embeddings()
# meta = load_omdb_meta()
#
# popular = pd.read_csv('/data/workspace/holly0015/test_project1/project1/src/data/parsed/movie_counts.csv')[:200]
# mov_base = {}
#
# pop_list = list(popular['id'])
#
# for i, k in list(meta.items()):
#      tmdid = int(meta[i]['tmdbId'])
#      if tmdid in pop_list:
#          print(tmdid)
#
#          movieid = pd.to_numeric(links.loc[tmdid]['movieId'])
#          if isinstance(movieid, pd.Series):
#              continue
#          mov_base[int(movieid)] = meta[i]['omdb']['Title']
#
# print(mov_base)
