import re
import math
import sys

class ProbSplit(object):
    def __init__(self, codes, separator='@@', vocab=None, glossaries=None):
        self.model = dict()
        self.wordsplit = dict()
        for line in codes:
            subunit = line.split()[0]
            prob = float(line.split()[1])
            self.model[subunit] = prob

    def segment(self, sentence):
        output = []
        for word in sentence.split():
            if word not in self.wordsplit:
                weight, splits = self._segment(word, 0, self.model)
                self.wordsplit[word] = splits
            output = output + self.wordsplit[word]
        return output

    def _segment(self, token, id, model):
        if id == len(token):
            #print id, 0
            return 0, []
        if id == len(token) - 1:
            #print id, -1
            return -10000000, []
        if id > len(token):
            assert(0)

        weight = []
        bestSplitCandidate = []
        for i in range(id + 2, min(len(token)+1, id + 6)):
            weight_rem, units = self._segment(token, i, model)
            bestunits = [token[id : i]] + units
            bestSplitCandidate.append(bestunits)
            weight.append(math.log(model.get(token[id : i], model['MIN' + str(i-id)])) + weight_rem)
        bestsplit = bestSplitCandidate[weight.index(max(weight))]
        return max(weight), bestsplit
