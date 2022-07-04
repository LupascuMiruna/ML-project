import numpy as np

class Reader:
    def upload(self, path):
        f = open(f"data_aliens/{path}_samples.txt", "r", encoding="utf8")
        lines =f.readlines()
        dictSamples = dict()
        #keep a dict
        #key = id
        #value = vect of words
        for line in lines:
            line = line.split("	", 1)
            dictSamples[line[0]] = line[1].strip()
        f.close()

        f = open(f"data_aliens/{path}_labels.txt", "r", encoding="utf8")
        lines = f.readlines()
        dictLabels = dict()
        # keep a dict
        # key = id
        # value = label
        for line in lines:
            line = line.split("	")
            dictLabels[line[0]] = line[1].strip()
        #keep just 2 vectors
        #labels[i] = the label for the vect of words from texts[i]
        texts = []
        labels = []
        for key in dictSamples:
            texts.append(dictSamples[key])
            labels.append(dictLabels[key])

        return texts, labels

    def uploadTest(self):
        f = open(f"data_aliens/test_samples.txt", "r", encoding="utf8")
        lines = f.readlines()
        dictSamples = dict()

        for line in lines:
            line = line.split("	", 1)
            dictSamples[line[0]] = line[1].strip()
        f.close()

        texts = []
        ids = []
        for key in dictSamples:
            texts.append(dictSamples[key])
            ids.append(key)

        return ids, texts