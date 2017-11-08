from collections import Counter
import numpy as np
import cPickle as cp
from constants import *

if DEBUG:
    TEXT = TINY_TEXT

word_freq = Counter()
data = filter(lambda x: len(x) != 1, open(TEXT).read().split(' '))
for word in data:
    word_freq[word] += 1
freq_list = word_freq.most_common()
word2ix = {PAD_TOK: 0}
min_freq = 5
for w, c in freq_list:
    if c < min_freq:
        break
    word2ix[w] = len(word2ix)
freq_table = np.zeros((len(word2ix), 1))
for w in word2ix:
    freq_table[word2ix[w]] = word_freq[w]
freq_table = freq_table ** 0.75
freq_table /= freq_table.sum()
unigram_table = []
table_length = 10e6
for w in word2ix:
    unigram_table += int(freq_table[word2ix[w]] * table_length) * [word2ix[w]]
unigram_table = np.array(unigram_table)

# Save the data
cp.dump(word2ix, open(VOCAB_FILE, "wb"))
np.save(UNIGRAM_TABLE_FILE, unigram_table)
