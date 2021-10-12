import json
import torch
import os
from fairseq.data.data_utils import collate_tokens
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import prince
from sklearn.preprocessing import OneHotEncoder
import itertools

import ppca
from ppca import PPCA

from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import pickle

"""

  1. torch + cuda
  pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

  2. torch scatter + torch cpu
  1.8.0+cpu
  pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cpu.html

  3. version check
  python -c "import torch; print(torch.__version__)"

"""

def data_embedding():
    thesis_list = pd.read_excel('thesis/parsed/thesis_list.xlsx')
    thesis_list = thesis_list[:10980]
    keyword_list = pd.read_excel('thesis/parsed/2021-07-01_scopus_WOS_keyword_list.xlsx')
    keyword_list['keyword'] = keyword_list.groupby(['thesis_ID'])['keyword'].transform(lambda x:' '.join(x))
    keyword_list = keyword_list.drop_duplicates()
    subject_list = pd.read_excel('thesis/parsed/2021-07-01_scopus_WOS_subject_list.xlsx')
    subject_list['subject_name'] = subject_list.groupby(['thesis_ID'])['subject_name'].transform(lambda x: ' '.join(x))
    subject_list = subject_list.drop_duplicates()

    thesis = pd.merge(thesis_list, keyword_list, left_on='article_DOI', right_on='thesis_ID', how='inner')
    thesis = pd.merge(thesis, subject_list, left_on='article_DOI', right_on='thesis_ID', how='inner')
    #keyword, subject_name, article_title, #Abstract
    #article_DOI, article_m
    thesis['article_m'] = thesis['article_title'] + thesis['keyword'] + thesis['subject_name']
    thesis = thesis[['movieId','article_m']].drop_duplicates().reset_index()


    #RoBERTa
    from fairseq.models.roberta import RobertaModel
    roberta = RobertaModel.from_pretrained('/data/workspace/holly0015/test_project1/roberta.base',
                                           checkpoint_file='model.pt')
    roberta.eval()

    fs = {}
    batch_size = 4

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ids = list(thesis['movieId'])
    plots = list(thesis['article_m'])
    plots = list(chunks(plots, batch_size))
    ids = list(chunks(ids, batch_size))

    def extract_features(batch, ids):
        batch = collate_tokens([roberta.encode(sent) for sent in batch], pad_idx=1)  # .to(cuda)
        batch = batch[:, :512]
        features = roberta.extract_features(batch)
        pooled_features = F.avg_pool2d(features, (features.size(1), 1)).squeeze()

        try:
            for i in range(pooled_features.size(0)):
                fs[ids[i]] = pooled_features[i].detach().cpu().numpy()
        except:
            print("____________________________ids")
            print(ids)
            print("____________________________pooled_features")
            print(pooled_features.size(0))


    for batch, ids in tqdm(zip(plots[::-1], ids[::-1]), total=len(plots)):
        extract_features(batch, ids)

    transformed = pd.DataFrame(fs).T
    # transformed.index = transformed.index.astype(int)
    # transformed = transformed.sort_index()

    print(transformed.head())

    #transformed.to_csv(os.path.join(os.getcwd(), "thesis/engineering/roberta.csv"), index=True, index_label='idx')
    # roberta = pd.read_csv(os.path.join(os.getcwd(), "data/engineering/roberta.csv"))

    #PCA Embedding
    roberta = transformed

    ppca = PPCA()
    ppca.fit(data=roberta.values.astype(float), d=128, verbose=False)
    ppca.var_exp

    transformed = ppca.transform()
    films_dict = dict(
        [(k, torch.tensor(transformed[i]).float()) for k, i in zip(roberta.index, range(transformed.shape[0]))])
    #pickle.dump(films_dict, open(os.path.join(os.getcwd(), "data/embeddings/ml20_pca128.pkl"), 'wb'))
    # embedding dict pkl dump

    pickle.dump(films_dict,
                open("/data/workspace/holly0015/test_project1/project1/src/thesis/embedding/thesis_pca128.pkl", 'wb'))

    return 0


    print(os.path.join(os.getcwd(), "data/parsed/omdb.json"))

    omdb = json.load(open(os.path.join(os.getcwd(), "data/parsed/omdb.json"),'r'))
    #tmdb = json.load(open("data/parsed/tmdb.json",'r'))
    #print(omdb['1'])

    #embedding nlp with RoBERTa

    batch_size = 4
    cuda = torch.device('cuda')
    # plots
    plots = []
    for i in omdb.keys():
        omdb_plot = omdb[i]['omdb'].get('Plot','')
        plot = omdb_plot
        plots.append((i, plot, len(plot)))

    plots = list(sorted(plots, key= lambda x:x[2]))
    plots = list(filter(lambda x : x[2] >4 , plots))
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    ids = [i[0] for i in plots]
    plots = [i[1] for i in plots]
    plots = list(chunks(plots, batch_size))
    ids = list(chunks(ids, batch_size))



    import urllib.request

    url = "https://korbillgates.tistory.com"
    res = urllib.request.urlopen(url)
    print(res.status)

    #https://pytorch.org/hub/pytorch_fairseq_roberta/
    #roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')#.to(cuda)
    # wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
    # tar -xzvf roberta.base.tar.gz
    from fairseq.models.roberta import RobertaModel
    roberta = RobertaModel.from_pretrained('/data/workspace/holly0015/test_project1/roberta.base', checkpoint_file='model.pt')
    roberta.eval()

    tokens = roberta.encode("Chicken Fries")
    print(tokens.tolist())
    print(roberta.decode(tokens))
    #Extract Features from RoBERTa
    print(roberta.extract_features(tokens))

    fs = {}
    def extract_features(batch, ids):
        batch = collate_tokens([roberta.encode(sent) for sent in batch], pad_idx = 1)#.to(cuda)
        batch = batch[:, :512]
        features = roberta.extract_features(batch)
        pooled_features = F.avg_pool2d(features, (features.size(1), 1)).squeeze()
        for i in range(pooled_features.size(0)):
            fs[ids[i]] = pooled_features[i].detach().cpu().numpy()

    for batch, ids in tqdm(zip(plots[::-1], ids[::-1]), total=len(plots)):
        extract_features(batch, ids)

    transformed = pd.DataFrame(fs).T
    transformed.index = transformed.index.astype(int)
    transformed = transformed.sort_index()

    print(transformed.head())
    print(transformed.size)

    transformed.to_csv(os.path.join(os.getcwd(), "data/engineering/roberta.csv"), index=True, index_label='idx')


    #Feature Embedding
    #MCA : Multiple Correspondence Aanlysis
    #omdb
    categorical = {
        'omdb': ['Rated', 'Director', 'Genre', 'Language', 'Country', 'Type', 'Production'],
    }

    def apply_categorical(records, type, take):
        res = {i: {} for i in records.keys()}
        for row in records.keys():
            for col in records[row][type].keys():
                if col in take:
                    res[row][col] = records[row][type][col]
        return res

    def apply_split(records, split, limit):
        for row in records.keys():
            for col in split:
                records[row][col] = tuple(records[row][col].split(', '))
        return records

    cat = apply_categorical(omdb, 'omdb', categorical['omdb'])
    cat = apply_split(cat, ['Country', 'Language', 'Genre'], 3)
    catdf = pd.DataFrame.from_dict(cat).T

    def one_hot(arr, name, categories):
        return dict((name + i, i in arr) for i in categories)

    def apply_one_hot(records, type, name, categories):
        for row in records.keys():
            records[row] = {**records[row], **one_hot(records[row][type], name, categories)}
            del records[row][type]
        return records

    genres_cat = list(set(itertools.chain(*tuple(catdf.Genre.unique()))))
    language_cat = pd.Series(list(itertools.chain(*catdf.Language))).value_counts()[:30].index
    countries_cat = pd.Series(list(itertools.chain(*catdf.Country))).value_counts()[:30].index

    cat = apply_one_hot(cat, 'Genre', 'g_', genres_cat)
    cat = apply_one_hot(cat, 'Country', 'c_', countries_cat)
    cat = apply_one_hot(cat, 'Language', 'l_', language_cat)

    catdf = pd.DataFrame.from_dict(cat).T
    catdf.Rated = catdf.Rated.fillna('Not rated')
    catdf.Rated[catdf.Rated == 'N/A'] = 'Not rated'
    catdf.Production.fillna('-')
    catdf.Production[catdf.Production == 'N/A'] = '-'
    catdf.Production[catdf.Production == 'NaN'] = '-'
    catdf.Production[catdf.Production.isna()] = '-'
    catdf.Director.fillna('-')
    catdf.Director[catdf.Director == 'N/A'] = '-'

    mca = prince.MCA(
        n_components=16,
        n_iter=20,
        copy=True,
        check_input=True,
        engine='auto',
    )
    mca = mca.fit(catdf)

    ax = mca.plot_coordinates(
        X=catdf,
        ax=None,
        figsize=(6, 6),
        show_row_points=True,
        row_points_size=10,
        show_row_labels=False,
        show_column_points=True,
        column_points_size=30,
        show_column_labels=False,
        legend_n_cols=1
    )

    ax.get_figure().savefig('data/images/mca_coordinates.svg')


    transformed = mca.transform(catdf)
    print(transformed.head())
    transformed.to_csv(os.path.join(os.getcwd(), "data/engineering/mca.csv"), index=True, index_label='idx')

    #PCA
    numerical = {
        'omdb': ['Year', 'Ratings', 'Metascore', 'imdbRating', 'imdbVotes'],
        'tmdb': ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    }

    def apply_numerical(records, type, take):
        res = {i: {} for i in records.keys()}
        for row in records.keys():
            for col in records[row][type].keys():
                if col in take:
                    res[row][col] = records[row][type][col]
        return res

    def apply_ratings(records):
        res = records.copy()
        for i in res.keys():
            for rating in res[i]['Ratings']:
                res[i]['r ' + rating['Source']] = rating['Value']
            del res[i]['Ratings']
        return res

    numo = apply_numerical(omdb, 'omdb', numerical['omdb'])
    num = dict([(i, {**numo[i]}) for i in numo.keys()])
    num = apply_ratings(num)

    numdf = pd.DataFrame.from_dict(num).T

    for col in numdf.columns:
        print(numdf.columns)
        numdf[col].loc[numdf[col] == 'N/A'] = np.nan
    # numdf['budget'] = numdf['budget'].replace(to_replace=0, value=np.nan)
    numdf['r Internet Movie Database'].loc[numdf['r Internet Movie Database'].notnull()] = \
        numdf['r Internet Movie Database'].loc[numdf['r Internet Movie Database'].notnull()].apply(
            lambda x: x.split('/')[0])
    numdf['r Metacritic'].loc[numdf['r Metacritic'].notnull()] = \
        numdf['r Metacritic'].loc[numdf['r Metacritic'].notnull()].apply(lambda x: int(x.split('/')[0]))
    numdf['r Rotten Tomatoes'].loc[numdf['r Rotten Tomatoes'].notnull()] = \
        numdf['r Rotten Tomatoes'].loc[numdf['r Rotten Tomatoes'].notnull()].apply(lambda x: float(x.replace('%', '')))
    #numdf['revenue'] = numdf['revenue'].replace(to_replace=0, value=np.nan)
    numdf['Year'].loc[numdf['Year'].notnull()] = numdf['Year'].loc[numdf['Year'].notnull()].apply(
        lambda x: int(x.replace('–', '')[0]))
    numdf['imdbVotes'].loc[numdf['imdbVotes'].notnull()] = numdf['imdbVotes'].loc[numdf['imdbVotes'].notnull()].apply(
        lambda x: int(x.replace(',', '')))

    print(numdf.head())

    ppca = PPCA()
    ppca.fit(data=numdf.values.astype(float), d=16, verbose=True)
    transformed = ppca.transform()
    transformed = pd.DataFrame(transformed)
    transformed['idx'] = pd.Series(list(omdb.keys()))
    transformed = transformed.set_index('idx')

    print(transformed.head())
    transformed.to_csv(os.path.join(os.getcwd(), "data/engineering/pca.csv"), index=True, index_label='idx')

    #Merge
    roberta = pd.read_csv(os.path.join(os.getcwd(), "data/engineering/roberta.csv"))
    cat = pd.read_csv(os.path.join(os.getcwd(), "data/engineering/mca.csv"))
    num = pd.read_csv(os.path.join(os.getcwd(), "data/engineering/pca.csv"))

    num = num.set_index('idx')
    cat = cat.set_index(cat.columns[0])
    roberta = roberta.set_index('idx')
    df = pd.concat([roberta, cat, num], axis=1)
    ppca = PPCA()
    ppca.fit(data=df.values.astype(float), d=128, verbose=True)

    transformed = ppca.transform()
    films_dict = dict(
        [(k, torch.tensor(transformed[i]).float()) for k, i in zip(df.index, range(transformed.shape[0]))])
    pickle.dump(films_dict, open(os.path.join(os.getcwd(), "data/embeddings/ml20_pca128.pkl"), 'wb'))
    #embedding dict pkl dump
    return films_dict