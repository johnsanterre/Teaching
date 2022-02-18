import numpy as np
import csv
from collections import Counter



with open('/Twitter/Sentiment Analysis Dataset.txt', 'r',  encoding="ISO-8859-1") as f:
  data = list(csv.reader(f, delimiter=','))

col_names = data[0]
data = data[1:]
data = [[int(x[0]),int(x[1]),[y.lower() for y in x[2].split()]] for x in data]

valid_chars = 'abcdefghijklmnopqrstuvwxyz'

for row  in data:
  tmp = []
  for word in row[2]:
    tmp.append(''.join([l for l in word if l in valid_chars]))
  row[2]=tmp

word_cnter = Counter([word for row in data for word in row[2]])

usable_words = [x[0] for x in word_cnter.most_common(200)]

for row in data:
  tmp=[]
  for word in row[2]:
    if word in usable_words:
      tmp.append(word)
  row[2]=tmp

for row in data:
  if len(row[2])>20:
    row[2]=row[2][-20:]
  else:
    tmp =[]
    for x in range(20-len(row[2])):
      tmp.append('a')
    tmp.extend(row[2])
    row[2]=tmp


