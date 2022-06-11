import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

data = [('professor', ['you', 'smell'], 1), 
        ('professor', ['you', 'fail'], 1), 
        ('professor', ['youre', 'bad'], 1),
        ('professor', ['above', 'average'], 0),
        ('professor', ['hate', 'you'], 1),
        ('professor', ['wiz', 'kid'], 0),
        ('professor', ['amazing', 'job'], 0),
        ('brother', ['great', 'job'], 1),
        ('brother', ['wiz', 'kid'], 1),
        ('brother', ['you', 'fail'], 0),
        ('brother', ['hate', 'you'], 0),
        ('brother', ['you', 'smell'], 0),
        ('mom', ['you', 'smell'], 0),
        ('mom', ['above', 'average'], 0),
        ('mom', ['you', 'bad'], 1),
        ('mom', ['love', 'you'], 0),
        ('mom', ['miss', 'you'], 0),
        ('mom', ['youre', 'disapointment'], 1),
        ('sister', ['amazing', 'job'], 1),
        ('sister', ['hate', 'you'], 0),
        ('sister', ['miss', 'you'], 1),
        ('sister', ['wiz', 'kid'], 1),
        ('sister', ['love', 'you'], 0),
        ('father', ['amazing', 'job'], 0),
        ('father', ['proud', 'you'], 0),
        ('father', ['work', 'harder'], 1),
        ('father', ['love', 'you'], 0),
        ('father', ['dont', 'quit'], 0)]

data_dev = [('professor', ['you', 'average'], 1), 
            ('brother', ['dont', 'quit'], 0),
            ('mom', ['nice', 'haircut'], 0),
            ('sister', ['bad', 'clothes'], 1),
            ('father', ['love', 'you'], 0)]
             

speakers, vocab = set(), set()
for i in range(len(data)):
  speakers.add(data[i][0])
  for j in range(CONTEXT_SIZE):
    vocab.add(data[i][1][j])

for i in range(len(data_dev)):
  speakers.add(data_dev[i][0])
  for j in range(CONTEXT_SIZE):
    vocab.add(data_dev[i][1][j])

word_to_ix = {word: i for i, word in enumerate(vocab)}
speaker_to_ix = {speaker: i for i, speaker in enumerate(speakers)}


class NGramLanguageModeler(nn.Module):
    def __init__(self, speaker_size, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings2 = nn.Embedding(speaker_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim+(context_size * embedding_dim), 128)
        self.linear2 = nn.Linear(128, 1)
    def forward(self, inputs):
        speaker,sentence = inputs
        sentence_embed = self.embeddings(sentence).view((1, -1))
        speaker_embed = self.embeddings2(speaker).view((1, -1))
        embeds_full = torch.cat((speaker_embed,sentence_embed), -1) 
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs


losses = []
val_losses = []
loss_function = nn.BCELoss()
model = NGramLanguageModeler(len(speakers), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    for speaker, sentence, target in data:
        word_idxs = [word_to_ix[w] for w in sentence]
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        speaker_idxs = [speaker_to_ix[speaker]]
        speaker_idxs = torch.tensor(speaker_idxs, dtype=torch.long)
        model.zero_grad()
        log_probs = model((speaker_idxs, word_idxs))
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.float).resize_((1, 1)))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    model.eval()
    with torch.no_grad():
      for speaker, sentence, target in data_dev:
        word_idxs = [word_to_ix[w] for w in sentence]
        word_idxs = torch.tensor(word_idxs, dtype=torch.long)
        speaker_idxs = [speaker_to_ix[speaker]]
        speaker_idxs = torch.tensor(speaker_idxs, dtype=torch.long)
        log_probs = model((speaker_idxs, word_idxs))
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.float).resize_((1, 1)))
        val_losses.append(loss.item())
    print('train_loss == ', losses[-1], 'dev_loss == ', val_losses[-1])



for row in zip(losses, val_losses):
  print(row)
