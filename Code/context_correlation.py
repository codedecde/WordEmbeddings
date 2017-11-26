class ContextCorr(object):
    def __init__(self, filenames, docname):
        """
        filenames: list of antonyms
        """
        self.antonymsPairs = list()
        self.words = list()
        for filename in filenames:
            self.antonymsPairs = self.antonymsPairs + [(line.split()[0], line.split()[1]) for line in open(filename)]
        for lid, line in enumerate(open(docname)):
            self.words = self.words + line.split()
        print 'Finished reading document'

    def calCorr(self):
        for pid, pair in enumerate(self.antonymsPairs):
            firstWord = pair[0]
            secondWord = pair[1]
            firstWordContext = dict()
            secondWordContext = dict()

            for wid, word in enumerate(self.words):
                if word == firstWord:
                    for i in range(wid - 4, wid + 5):
                        contextWord = self.words[i]
                        firstWordContext[contextWord] = firstWordContext.get(contextWord, 0) + 1
                if word == secondWord:
                    for i in range(wid - 4, wid + 5):
                        contextWord = self.words[i]
                        secondWordContext[contextWord] = secondWordContext.get(contextWord, 0) + 1

            correlation = self.innerProd(firstWordContext, secondWordContext)
            print pid, ',', firstWord, ',', secondWord, ',', correlation
            #if pid > 10: break

    @staticmethod
    def innerProd(firstWordContext, secondWordContext):
        firstWordContextCount = float(sum(firstWordContext.values()))
        secondWordContextCount = float(sum(secondWordContext.values()))
        dotProd = 0
        for word, count in firstWordContext.items():
            dotProd += count * secondWordContext.get(word, 0) / firstWordContextCount / secondWordContextCount

        return dotProd

if __name__ == "__main__":
    filenames = ['../Data/Linguistic_Constraints/wordnet_antonyms.txt']
    docname = '../Data/text8'
    cc = ContextCorr(filenames, docname)
    cc.calCorr()
