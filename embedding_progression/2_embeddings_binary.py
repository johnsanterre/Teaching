#  https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
# lightly modified from above

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

data = [(['you', 'stink'], 1),
        (['you', 'amazing'], 0)]

vocab = set([y for x in data for y in x[0]])
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1)
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.sigmoid(out)
        return log_probs


losses = []
loss_function = nn.BCELoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in data:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor(np.array([[target]]).reshape((1,1)), dtype=torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)

print(losses)
