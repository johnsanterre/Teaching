import csv
import numpy as np
import random
import matplotlib.pyplot as plt

#how to read a cvs from git hub
# Swap to github data loc here: https://raw.githubusercontent.com/crwong/cs224u-project/master/data/sentiment/training.1600000.processed.noemoticon.csv

with open("/trainingandtestdata/training.1600000.processed.noemoticon.csv", encoding='utf-8') as f:
    data = list(csv.reader(f))

data2 = [[x[4],x[5].lower(),x[0]] for x in data if x[0]=='0' or x[0]=='4']
random.shuffle(data2)

total_words = set(y for x in data2 for y in x[1].split())

from collections import Counter
negative_words = [y for x in data2 for y in x[1].split() if x[2]=='0']
#Counter(negative_words).most_common(100)
top_neg = [x[0] for x in Counter(negative_words).most_common(20000)]



positive_words = [y for x in data2 for y in x[1].split() if x[2]=='4']
top_pos =[x[0] for x in  Counter(positive_words).most_common(20000)]

len_neg_sentence = [len(x[1].split()) for x in data2  if x[2]=='0']
len_pos_sentence = [len(x[1].split()) for x in data2  if x[2]=='4']

pos_words = [x for x in top_pos if x not in top_neg]
neg_words = [x for x in top_neg if x not in top_pos]


ret =[]
for row in data2[:100000]:
    pos_cnt = 0
    neg_cnt = 0
    for word in row[1].split():
        if word in pos_words:
            pos_cnt +=1
        elif word in neg_words:
            neg_cnt -=1
    if (pos_cnt+neg_cnt)==0:
        ret.append(0)
    else:
        ret.append(pos_cnt+neg_cnt)

        
plt.hist([x for idx, x in enumerate(ret) if x!=0 and data2[idx][2]=='4'], bins =100)
plt.hist([x for idx, x in enumerate(ret) if  x!=0 and data2[idx][2]=='0'], bins =100)
plt.show()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=.99, min_df=.00025)
X = vectorizer.fit_transform([x[1] for x in data2])

X.shape

vocab = vectorizer.vocabulary_
reverse_vocab = dict([v,k] for k,v in vocab.items())

####
postive_vec = np.zeros(X.shape[1])
negative_vec = np.zeros(X.shape[1])
for idr, row in enumerate(data2):
    if row[2]=='0':
        negative_vec+=X[idr,:]
    else:
        postive_vec+=X[idr,:]

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(negative_vec,postive_vec)

np.max(postive_vec-negative_vec)
np.where(np.max(postive_vec-negative_vec)==postive_vec-negative_vec)
reverse_vocab=dict([v,k] for k,v in vocab.items())
reverse_vocab[np.where(np.max(postive_vec-negative_vec)==postive_vec-negative_vec)[1][0]]
reverse_vocab[np.where(np.max(negative_vec-postive_vec)==negative_vec-postive_vec)[1][0]]
####

L = [0 if x[2]=='0' else 1 for x in data2]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NN_structure = [(300,100), (100,100), (100,1)]
class Modeler(nn.Module):
    def __init__(self, tfidf_size, NN_structure):
        super(Modeler,self).__init__()
        self.linear_tfidf = nn.Linear(tfidf_size, 300)
        self.linears = nn.ModuleList([nn.Linear(*x) for x in NN_structure])
    def forward(self, tfidf):
        tfidf_out = F.relu(self.linear_tfidf(tfidf))
        for layer in self.linears[:-1]:
            tfidf_out = F.relu(layer(tfidf_out))
        return torch.sigmoid(self.linears[-1](tfidf_out))

tfidf_size = X.shape[1]
batch_size = 128

model = Modeler(tfidf_size, NN_structure)
device = torch.device('cpu')
model.to(device)
loss_function = nn.BCELoss()
optmizer = optim.Adam(model.parameters(), lr=.0001)

for epoch in range(20):
    print(epoch)
    total_loss,val_loss = [],[]
    for idx in range(((X.shape[0]//batch_size)-1)):
        model.zero_grad()
        start = (idx*batch_size)
        stop = start+batch_size
        probs = model(torch.tensor(X[start:stop,].toarray(), dtype=torch.float).to(device))
        loss = loss_function(probs,torch.tensor(L[start:stop],dtype=torch.float).reshape((batch_size,1)).to(device))
        loss.backward()
        optmizer.step()
        total_loss.append(loss.item())
        print(np.mean(total_loss))

sentences = []
for row in data2:
    tmp = np.zeros(25)
    for idw,word in enumerate(row[1].split()[:25]):
        if word in vocab:
            tmp[idw]=vocab[word]
    sentences.append(tmp)

sentences = np.array(sentences)

NN_structure = [(10,100), (100,1)]
class Modeler(nn.Module):
    def __init__(self, tfidf_size, NN_structure):
        super(Modeler,self).__init__()
        self.embeddings_words = nn.Embedding(len(vocab),10)
        self.linears = nn.ModuleList([nn.Linear(*x) for x in NN_structure])
    def forward(self,words):
        words_embeds = torch.sum(self.embeddings_words(words), axis=1)
        for layer in self.linears[:-1]:
            words_embeds = F.relu(layer(words_embeds))
        return torch.sigmoid(self.linears[-1](words_embeds))

tfidf_size = X.shape[1]
batch_size = 128

model = Modeler(tfidf_size, NN_structure)
device = torch.device('cpu')
model.to(device)
loss_function = nn.BCELoss()
optmizer = optim.Adam(model.parameters(), lr=.0001)

for epoch in range(20):
    print(epoch)
    total_loss,val_loss = [],[]
    for idx in range(((X.shape[0]//batch_size)-1)):
        model.zero_grad()
        start = (idx*batch_size)
        stop = start+batch_size
        probs = model(torch.tensor(X[start:stop,].toarray(), dtype=torch.int).to(device))
        loss = loss_function(probs,torch.tensor(L[start:stop],dtype=torch.float).reshape((batch_size,1)).to(device))
        loss.backward()
        optmizer.step()
        total_loss.append(loss.item())
        print(np.mean(total_loss))
