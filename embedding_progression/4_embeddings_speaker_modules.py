import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)
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

speakers, vocab = set(), set()
for i in range(len(data)):
  speakers.add(data[i][0])
  for j in range(CONTEXT_SIZE):
    vocab.add(data[i][1][j])

word_to_ix = {word: i for i, word in enumerate(vocab)}
speaker_to_ix = {speaker: i for i, speaker in enumerate(speakers)}

class Speaker(nn.Module):
  def __init__(self, speaker_size, embedding_dim):
    super(Speaker, self).__init__()
    self.speaker_embeddings = nn.Embedding(speaker_size, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim, 32)
    self.linear2 = nn.Linear(32, 32)
    self.linear3 = nn.Linear(32, 64)
  def forward(self,inputs):
    speaker = inputs
    speaker_embed = self.speaker_embeddings(speaker).view((1, -1))
    out = F.relu(self.linear1(speaker_embed))
    out = F.relu(self.linear2(out))
    out = F.relu(self.linear3(out))
    return out

class Sentence(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size):
    super(Sentence, self).__init__()
    self.sentence_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(embedding_dim*2, 32)
    self.linear2 = nn.Linear(32, 32)
    self.linear3 = nn.Linear(32, 64)
  def forward(self,inputs):
    sentence = inputs
    sentence_embed = self.sentence_embeddings(sentence).view((1, -1))
    out = F.relu(self.linear1(sentence_embed))
    out = F.relu(self.linear2(out))
    out = F.relu(self.linear3(out))
    return out

class MainModule(nn.Module):
    def __init__(self, speaker_size, vocab_size, embedding_dim, context_size):
        super(MainModule, self).__init__()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 1)
        self.speaker_module = Speaker(speaker_size, embedding_dim)
        self.sentence_module = Sentence(vocab_size, embedding_dim, context_size)
    def forward(self, inputs):
        speaker,sentence = inputs
        sentence_embed = self.sentence_module(sentence).view((1, -1))
        speaker_embed = self.speaker_module(speaker).view((1, -1))
        embeds_full = torch.cat((speaker_embed,sentence_embed), -1) 
        out = F.relu(self.linear1(embeds_full))
        out = self.linear2(out)
        log_probs = torch.sigmoid(out)
        return log_probs


losses = []
loss_function = nn.BCELoss()
model = MainModule(len(speakers), len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
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


print(losses)  # The loss decreased every iteration over the training data!
