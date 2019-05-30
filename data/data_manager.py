
import os, json
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from data.data_sets import FolderDataset
from utils.util import list_dir


def decode_json(dic):
    ret = {}
    for k, v in dic.items():
        ret[k] = {} 
        for l, m in v.items():
            ret[k][int(l)] = m
    return ret

class CSVDataManager(object):

    def __init__(self, config):

        self.dir_path = config['path']
        self.splits = config['splits']
        self.loader_params = config['loader']
        join_dpath = lambda x: os.path.join(self.dir_path, x)
        self.path_movies = join_dpath('movies.dat')
        self.path_ratings = join_dpath('ratings.dat')

        if os.path.isfile(join_dpath('.mappings.json')):
            self.data = {}
            for k in self.splits.keys():
                self.data[k] = pd.read_csv(join_dpath('.%s_data.txt'%k), sep='\t')
            with open(join_dpath('.mappings.json'), 'r') as jr:
                mappings = json.load(jr)
        
            self.mappings = decode_json(mappings)   

        else:
            all_data, self.mappings = self.process_dataset()

            with open(join_dpath('.mappings.json'), 'w') as jw:
                json.dump(self.mappings, jw)

            self.data = self.split_data(all_data)
            for k, v in self.data.items():
                v.to_csv(join_dpath('.%s_data.txt'%k), index=None, sep='\t')

        self.dims = self.get_dims()
        
    def get_dims(self):
        ret = []
        for df in self.data.values():
            ret.append(df['users'])

        return {    'users'   : pd.concat(ret).nunique(),
                    'generes' : len(self.mappings['generes']),
                    'movies'  : max(self.mappings['movies'].keys()) + 1}

    def process_dataset(self):

        def _process_movies(df):
            def _genere_map(col):
                ret = []
                for a in col:
                    ret += a.split('|')
                generes = np.unique(ret)
                # leave index 0 for no category!
                inds = list(range(1, len(generes)+1))
                return dict(zip(inds, generes))

            def _movie_map(df, g_r):
                g = {v:k for k, v in g_r.items()}
                
                df['generes'] = df['generes'].apply(lambda x: [g[a] for a in x.split('|')])
                movie_to_genere = {int(k):v for k, v in df[['movies', 'generes']].values}                
                movies = {int(k):v for k, v in df[['movies', 'name']].values}
                return movies, movie_to_genere
            
            df.columns = ['movies', 'name', 'generes']
            df['movies'] -= 1

            generes = _genere_map(df['generes'])
            movies, movie_to_genere = _movie_map(df, generes)
            return movies, generes, movie_to_genere

        def _process_ratings(df, genere_map):

            df.columns = ['users', 'movies', 'rating', 'ts']
            df['users'] -= 1
            df['movies'] -= 1
            df = df.drop('ts', 1)
            df['generes'] = df['movies'].apply(lambda x: ','.join([str(y) for y in genere_map[x]]))
            return df[['users', 'movies', 'generes', 'rating']]

        df_movies = pd.read_csv(self.path_movies, sep='::', header=None, engine='python') 
        movies, generes, movie_to_genere = _process_movies(df_movies)

        df_ratings = pd.read_csv(self.path_ratings, sep='::', header=None, engine='python') 
        data = _process_ratings(df_ratings, movie_to_genere)
        mappings = {'movies':movies, 
                    'generes':generes, 
                    'movie_to_genere':movie_to_genere
                    }
        return data, mappings


    def split_data(self, all_data):
        size = len(all_data)
        inds = np.arange(size)
        np.random.shuffle(inds)
        
        left = 0
        ret = {}
        for k,v in self.splits.items():
            right = left + int(round(size*v))
            ret[k] = all_data.iloc[left:right]
            left = right
        return ret

    def get_loader(self, name):
        assert name in self.data
        dataset = FolderDataset(self.data[name])
        return data.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        users, items, gens, targets = zip(*batch)

        users, items, targets = map(torch.LongTensor, [users, items, targets])
        gens = list(map(torch.LongTensor, gens))

        gens = torch.nn.utils.rnn.pad_sequence(gens, batch_first=True, padding_value=0)

        return users, items, gens, targets.float()

if __name__ == '__main__':

    pass




