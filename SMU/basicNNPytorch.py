#torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

XX = np.array([[0,1,0],
               [1,1,0],
               [0,0,0],
               [1,0,0]]) 
LL = np.array([0,1,1,0])

XX_dev = np.array([[0,1,1],
                  [1,1,0],
                  [0,0,1],
                  [1,0,0]]) 
LL_dev = np.array([0,1,1,0]) 


batch_size = 2
NN_structure = [(XX.shape[1], 16),(16,32),(32,1)] 

class Modeler(nn.Module):
    def __init__(self ): 
        super(Modeler, self).__init__() 
        self.linears = nn.ModuleList([nn.Linear(*x) for x in NN_structure])
        #self.linear1 = nn.Linear(XX.shape[1], 16)
        #self.linear2 = nn.Linear(16,32)
        #self.linear3 = nn.Linear(32,1)
    def forward(self, rows):
        for layer in self.linears[:-1]:
          rows = F.relu(layer(rows))
        rows = torch.sigmoid(self.linears[-1](rows))
        #rows = F.relu( self.linear1(rows))
        #rows = F.relu( self.linear2(rows))
        #rows = torch.sigmoid( self.linear3(rows))
        return rows

model = Modeler()
loss_function = nn.BCELoss() # often can change!
optimizer = optim.Adam(model.parameters(), lr=0.01)

tr_loss = []
validation_loss=[]

for epoch in range(3):
  tmp_loss = []
  for idx in range(len(XX)//batch_size):
    model.zero_grad() # EVERY TIME
    start = idx*batch_size
    stop = idx*batch_size+batch_size
    probs = model(torch.tensor(XX[start:stop,], dtype=torch.float))# EVERY TIME
    loss = loss_function(probs, torch.tensor(LL[start:stop,],dtype=torch.float).reshape((batch_size,1)))# EVERY TIME
    loss.backward() # EVERY TIME
    optimizer.step()
    tmp_loss.append(loss.item())
  tr_loss.append(np.mean(tmp_loss))
  model.eval() # EVERY TIME
  with torch.no_grad(): # EVERY TIME
    tmp_val_loss = []
    for idx in range(len(XX_dev)//batch_size):
      start = idx*batch_size
      stop = idx*batch_size+batch_size
      probs = model((torch.tensor(XX_dev[start:stop,], dtype=torch.float)))
      loss = loss_function(probs, torch.tensor(LL_dev[start:stop,],dtype=torch.float).reshape((batch_size,1)))
      tmp_val_loss.append(loss.item())
    validation_loss.append(np.mean(tmp_val_loss))

plt.plot(tr_loss, color='red')
plt.plot(validation_loss)
plt.show()
import pdb; pdb.set_trace()
