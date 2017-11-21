import sys
import re
import collections
import math

def train(trainfile, modelfile, threshold):
	gramLengths = [2,3,4,5]
	#grams = dict.fromkeys(gramLengths, dict())
	grams = {key: {} for key in gramLengths}
	f = open(trainfile)
	for line in f:
		tokens = line.split()
		for token in tokens:
			token = 'B' + token + 'E'
			for n in gramLengths:
				for i in range(len(token) - n + 1):
					substr = token[i : i + n]
					grams[n][substr] = grams[n].get(substr, 0) + 1
	
	with open(modelfile, 'w') as fout:
		model = dict()
		for n, dic in grams.items():
			total = sum(dic.values())
			print '# total keys:', n, len(dic.keys())
			print '# filtered keys: ', sum(1 for i in dic.values() if i / float(total) > threshold)
			fout.write('MIN' + str(n) + '\t' + str(1 / float(total)) + '\n')
			model['MIN' + str(n)] = 1 / float(total)

			for key, value in dic.items():
				prob = value / float(total)
				if  prob> threshold:
					model[key] = prob
					fout.write(key + '\t' + str(prob) + '\n')

	return model

def evaluate(model, testfile):
	filename = sys.argv[3]
	f = open(filename)
	for line in f:
		tokens = line.split()
		for token in tokens:
			token = 'B' + token + 'E'
			weight, units = segment(token, 0, model)
			print token, units

def segment(token, id, model):
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
		weight_rem, units = segment(token, i, model)
		bestunits = [token[id : i]] + units
		bestSplitCandidate.append(bestunits)
		weight.append(math.log(model.get(token[id : i], model['MIN' + str(i-id)])) + weight_rem)
	bestsplit = bestSplitCandidate[weight.index(max(weight))]
	return max(weight), bestsplit

def loadmodel(modelfile):
	model = dict()
	with open(modelfile) as f:
		for line in f:
			tokens = line.split()
			model[tokens[0]] = float(tokens[1])
	return model


if __name__ == '__main__':
	threshold = 1e-5
	trainfile = sys.argv[1]
	modelfile = sys.argv[2]
	#testfile = sys.argv[3]
	model = train(trainfile, modelfile, threshold)
	# load pre-trained model
	model = loadmodel(modelfile)
	print segment('BexcitingE', 0, model)




