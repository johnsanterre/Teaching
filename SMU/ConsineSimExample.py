import numpy as np
from  scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


data = np.array([[.9,0,0],
                 [1,.1,.4],
                 [1.2,.13,.1],
                 [.1,1.2,.3],
                 [.2,1.1,.1],
                 [0,.9,.2],
                 [0,.1,2],
                 [.2,.3,1],
                 [0.2,0.3,1.3]])

L = np.array([1,1,1,2,2,2,3,3,3])

avg_vectors = []
for label in [1,2,3]:
  tmp = np.zeros(data.shape[1])
  for entry in  np.where(L==label)[0]:
    tmp+=data[entry,:]
  avg_vectors.append(tmp)

M = np.array(avg_vectors)

ret=dict()
for x in range(M.shape[0]):
  for y in range(M.shape[0]):
    if x<y:
      if (x,y) not in ret:
        ret[(x,y)]= 1-cosine(M[x,:], M[y,:])

