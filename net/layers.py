
import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstOrder(nn.Module):

    def __init__(self, dims):
        super(FirstOrder, self).__init__()

        self.bias_user = nn.Embedding(dims['users'], 1)
        self.bias_movie = nn.Embedding(dims['movies'], 1)
        # Add 1 to allow for padding idx
        self.bias_genere = nn.Embedding(dims['generes'] + 1, 1, padding_idx=0)

    def forward(self, batch):
        users, movies, gens = batch

        b_u = self.bias_user(users)
        b_i = self.bias_movie(movies)
        b_g = self.bias_user(gens).sum(-2)

        return (b_u + b_i + b_g).squeeze()


class SecondOrder(nn.Module):

    def __init__(self, dims, k):
        super(SecondOrder, self).__init__()

        self.emb_user = nn.Embedding(dims['users'], k)
        self.emb_movie = nn.Embedding(dims['movies'], k)
        # Add 1 to allow for padding idx
        self.emb_genere = nn.Embedding(dims['generes'] + 1, k, padding_idx=0)

    def _get_stacked_embedded(self, users, movies, gens):
        v_u = self.emb_user(users).unsqueeze(1)
        v_i = self.emb_movie(movies).unsqueeze(1)
        v_g = self.emb_user(gens)

        return torch.cat([v_u, v_i, v_g], dim=1)

    def forward(self, batch):
        users, movies, gens = batch
        v = self._get_stacked_embedded(users, movies, gens)

        ret = torch.zeros(v.size(0)).to(v.device)
        for i in range(v.size(1)):
            for j in range(i + 1, v.size(1)):
                ret += (v[:, i] * v[:, j]).sum(-1)
        return ret


class Attention(nn.Module):

    def __init__(self, k, t):
        super(Attention, self).__init__()
        self.lin = nn.Linear(k, t)
        self.h = nn.Parameter(torch.rand(t, 1))

    def forward(self, x):
        return F.relu(self.lin(x)).mm(self.h)


class AttentionSecondOrder(SecondOrder):

    def __init__(self, dims, k, t):
        super(AttentionSecondOrder, self).__init__(dims, k)
        self.p = nn.Parameter(torch.rand(k, 1))
        self.att = Attention(k, t)

    def forward(self, batch):
        users, movies, gens = batch
        v = self._get_stacked_embedded(users, movies, gens)

        ret = torch.zeros(v.size(0), v.size(-1)).to(v.device)

        e = []
        sumatori = []
        for i in range(v.size(1)):
            for j in range(i + 1, v.size(1)):
                elem_wise = v[:, i] * v[:, j]

                e.append(self.att(elem_wise))
                sumatori.append(elem_wise)

        alphas = F.softmax(torch.cat(e, -1), 1)
        for i in range(len(sumatori)):
            ret += alphas[:, i:i + 1] * sumatori[i]

        return (ret.mm(self.p)).squeeze()
