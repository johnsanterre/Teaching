# https://grouplens.org/datasets/movielens/


#https://en.wikipedia.org/wiki/QR_decomposition

import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg
from sklearn.decomposition import NMF


# Load data
with open('ratings.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data[0]
users = list(set([int(x[0]) for x in data[1:]]))
movies = list(set([int(x[1]) for x in data[1:]]))

M = np.zeros((len(users),len(movies)))

for x in data[1:]:
    M[users.index(int(x[0])),movies.index(int(x[1]))]= float(x[2])
import pdb; pdb.set_trace()

### NMF
model = NMF(n_components=(100))
ft = model.fit_transform(M)

linalg.norm(np.dot(ft, model.components_)-M)


ret = []
for x in range(5,10):
    model = NMF( n_components=(5*x), init='random', max_iter=1000)
    ft = model.fit_transform(M)
    ret.append(linalg.norm(np.around(np.dot(ft, model.components_),decimals=1)-M))

plt.plot(ret)
plt.show()


import pdb; pdb.set_trace()
### SVD
M_normed = np.mean(M, axis=1)
M_demeaned  = M - M_normed.reshape(-1,1)
U, S, V = np.linalg.svd(M_demeaned)
M_projected=np.dot(np.dot(U[:,:3],np.diag(S[:3])),V[:3,:])
M_projected+ M_normed.reshape(-1,1)

ret = []
for x in range(61):
    M_projected=np.dot(np.dot(U[:,:x*10],np.diag(S[:x*10])),V[:x*10,:])
    ret.append(linalg.norm((M_projected + M_normed.reshape(-1,1))-M))

plt.plot(ret)
plt.show()

import pdb; pdb.set_trace()


#Rank Aggregation
M_rank = np.zeros((M.shape[1],M.shape[1]))

#Why not this?????
#for idx, row in enumerate(M):
#  print(idx)
#  for idr, r in enumerate(row[:-1]):
#    for idy, y in enumerate(row[idr+1:]):
#      M_rank[idr,idy] += (r-y)
#      M_rank[idy,idr] -= (r-y)
#
compressed_size = 1000

ret = np.zeros((compressed_size, compressed_size))
for z in range(M.shape[0]):
    print(z)
    ret += np.dot(M[z,:compressed_size].reshape(-1,1),np.ones((compressed_size)).reshape(-1,1).T) - np.dot(np.ones((compressed_size)).reshape(-1,1), M[z,:compressed_size].reshape(-1,1).T)

MM = np.exp(ret/ret.max())

x = np.ones((MM.shape[0],1))
for _ in range(100):
    x = np.dot(MM,x)

ddd = x.T.argsort()[0]




#############
model = NMF(n_components=(40), init='random', max_iter=1000)

ft = model.fit_transform(M)
ft.shape
model.components_.shape

####
np.around(np.dot(ft, model.components_),decimals=1)
####
linalg.norm(np.around(np.dot(ft, model.components_),decimals=1)-M)
###
Counter(zip([y for x in M for y in x], [y for x in np.round(np.dot(ft, model.components_)) for y in x]))
###
ret = []
for x in range(5,10):
    model = NMF( n_components=(5*x), init='random', max_iter=1000)
    ft = model.fit_transform(M)
    #zz  = model.inverse_transform(ft)
    #linalg.norm(zz-M)
    ret.append(linalg.norm(np.around(np.dot(ft, model.components_),decimals=1)-M))
plt.plot(ret)
plt.show()
####

####
model = NMF( n_components=(50), init='random', max_iter=1000)
ft = model.fit_transform(M)
Counter(zip([y for x in M for y in x], [y for x in np.round(np.dot(ft, model.components_)) for y in x]))

plt.matshow( model.components_[:,:200])
plt.show()

Counter([ sum(model.components_[x]>.1) for x in range(100)])


import pdb; pdb.set_trace()





import pdb; pdb.set_trace()