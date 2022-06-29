import csv
from collections import Counter

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


with open('/Users/john/Desktop/creditcard.csv') as f:
    data = list(csv.reader(f))

col_names = data[0]
data = data[1:]

# Counter([x[-1] for x in data])

train, test = train_test_split(data, test_size=.2, random_state=314)

# Counter([x[-1] for x in train])
# Counter([x[-1] for x in test])

train_major = [x for x in train if x[-1]=='0']
train_minor = [x for x in train if x[-1]=='1']

train_minor_up = resample(train_minor, replace = True, n_samples = len(train_major), random_state=314)

train_up = np.array(train_major + train_minor_up, dtype='float64')

test_major = [x for x in test if x[-1]=='0']
test_minor = [x for x in test if x[-1]=='1']

test_minor_up = resample(test_minor, replace = True, n_samples = len(test_major), random_state=314)

test_up = np.array(test_major + test_minor_up, dtype='float64')

clf = RandomForestClassifier()
clf.fit(train_up[:,:-1], train_up[:,-1])

confusion_up = Counter(zip(clf.predict(test_up[:,:-1]),test_up[:,-1]))
accuracy_up =( confusion_up[(0,1)]+confusion_up[(1,0)])/len(test_up)

test = np.array(test,dtype='float64')
confusion = Counter(zip(clf.predict(test[:,:-1]),test[:,-1]))
accuracy = (confusion[(0,1)]+confusion[(1,0)])/len(test_up)

