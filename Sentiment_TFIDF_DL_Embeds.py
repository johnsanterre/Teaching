import csv
import numpy as np
import random

#data is here
#http://help.sentiment140.com/for-students

#https://medium.com/code-kings/python3-fix-unicodedecodeerror-utf-8-codec-can-t-decode-byte-in-position-be6c2e2235ee
with open('training.1600000.processed.noemoticon.csv','rt',encoding='utf-8') as f:
  data = list(csv.reader(f))

data2 = [[x[4],x[5].lower(), x[0]] for x in data if x[0]=='0' or x[0]=='4']
random.shuffle(data2)

total_words = set(y for x in data2 for y in x[1].split())
from collections import Counter
negative_words = [y for x in data2 for y in x[1].split() if x[2]=='0']
Counter(negative_words).most_common(100)

positive_words = [y for x in data2 for y in x[1].split() if x[2]=='4']
Counter(positive_words).most_common(100)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words ='english', max_df=0.99, min_df=0.00025)
X = vectorizer.fit_transform([x[1] for x in data2])

X.shape

vocab  = vectorizer.vocabulary_
reverse_vocab = dict([v,k] for k,v in vocab.items())

####
positive_vec = np.zeros(X.shape[1])
negative_vec = np.zeros(X.shape[1])
for idr, row in enumerate(data2):
  if row[2]=='0':
    negative_vec+=X[idr,:]
  else:
    positive_vec+=X[idr,:]

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(negative_vec,positive_vec)

np.max(positive_vec-negative_vec)
np.where(np.max(positive_vec-negative_vec)==positive_vec-negative_vec)
reverse_vocab[np.where(np.max(positive_vec-negative_vec)==positive_vec-negative_vec)[1][0]]
reverse_vocab[np.where(np.max(negative_vec-positive_vec)==negative_vec-positive_vec)[1][0]]
####

L =[0 if x[2]=='0' else 1 for x in data2]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NN_structure = [(3000, 100),(100, 100),(100, 100),(100, 100),(100, 1)] 
class Modeler(nn.Module):
    def __init__(self, tfidf_size,  NN_structure): 
        super(Modeler, self).__init__()
        self.linear_tfidf = nn.Linear(tfidf_size, 3000)
        self.linears = nn.ModuleList([nn.Linear(*x) for x in NN_structure])
    def forward(self, tfidf):
        tfidf_out = F.relu(self.linear_tfidf(tfidf))
        for layer in self.linears[:-1]:
          tfidf_out  = F.relu(layer(tfidf_out))
        return torch.sigmoid(self.linears[-1](tfidf_out)) 

tfidf_size = X.shape[1]
batch_size = 128

model = Modeler(tfidf_size,  NN_structure)
#device = torch.device("cuda:0")
device = torch.device('cpu')  
model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(20):
  print(epoch)
  total_loss,val_loss = [],[]
  for idx in range((((X.shape[0])//batch_size) -1)):
    model.zero_grad()
    start = (idx*batch_size) 
    stop = start+batch_size
    probs = model(torch.tensor(X[start:stop,].toarray(), dtype=torch.float).to(device))
    loss = loss_function(probs, torch.tensor(L[start:stop],dtype=torch.float).reshape((batch_size,1)).to(device))
    loss.backward()
    optimizer.step()
    total_loss.append(loss.item())
    print(np.mean(total_loss))

#######



sentences = []
for row in data2:
  tmp = np.zeros(25)
  for idw,word in enumerate(row[1].split()[:25]):
    if word in vocab:
      tmp[idw]=vocab[word]
  sentences.append(tmp)

sentences= np.array(sentences)

NN_structure = [(10, 100),(100, 1)] 
class Modeler(nn.Module):
    def __init__(self, tfidf_size,  NN_structure): 
        super(Modeler, self).__init__()
        self.embeddings_words = nn.Embedding(len(vocab), 10)
        self.linears = nn.ModuleList([nn.Linear(*x) for x in NN_structure])
    def forward(self, words):
        words_embeds = torch.sum(self.embeddings_words(words), axis=1)
        for layer in self.linears[:-1]:
          words_embeds  = F.relu(layer(words_embeds))
        return torch.sigmoid(self.linears[-1](words_embeds)) 

tfidf_size = X.shape[1]
batch_size = 128

model = Modeler(tfidf_size,  NN_structure)
device = torch.device("cuda:0")
device = torch.device('cpu')  
model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(20):
  print(epoch)
  total_loss,val_loss = [],[]
  for idx in range((((X.shape[0])//batch_size) -1)):
    model.zero_grad()
    start = (idx*batch_size) 
    stop = start+batch_size
    probs = model(torch.tensor(sentences[start:stop,], dtype=torch.int).to(device))
    loss = loss_function(probs, torch.tensor(L[start:stop],dtype=torch.float).reshape((batch_size,1)).to(device))
    loss.backward()
    optimizer.step()
    total_loss.append(loss.item())
    print(np.mean(total_loss))

