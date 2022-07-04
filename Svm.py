import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt
from Reader import Reader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas

class SVM:
    def __init__(self, model, vectorizer):
        self.reader = Reader()

        train, training_labels = self.reader.upload("train")
        validation, validation_labels = self.reader.upload("validation")
        test_ids, test_sentences = self.reader.uploadTest()

        self.train = train # vectors of words
        self.training_labels = training_labels
        self.validation = validation
        self.validation_labels = validation_labels
        self.test_ids = test_ids
        self.test_sentences = test_sentences
        self.train_validation_labels = training_labels + validation_labels #concatenate for a bigger input for predictions

        self.training_matrix = pandas.DataFrame() #labeled dictionary as a matrix
        self.training_matrix['text'] = train
        self.training_matrix['label'] = training_labels

        self.model = model
        self.vectorizer = vectorizer

    def trainModel(self):
        self.vectorizer.fit(self.training_matrix['text'])

    def getVectors(self):
        train_vect = self.vectorizer.transform(self.train)
        valid_vect = self.vectorizer.transform(self.validation)
        test_vect = self.vectorizer.transform(self.test_sentences)
        train_valid_vect = self.vectorizer.transform(self.train + self.validation)
        return train_vect, valid_vect, test_vect, train_valid_vect

    def predictLanguage(self):
        self.trainModel()
        train_vect, valid_vect, test_vect, train_valid_vect = self.getVectors()
        self.model.fit(train_valid_vect, self.train_validation_labels) # refit the model for a bigger input
        predictions = self.model.predict(test_vect) # predict the labels
        for i in range(len(self.test_ids)):
            print(self.test_ids[i], predictions[i], sep=",")

    def confusionMatrix(self, predicted_labels):
        confusionMatrix = confusion_matrix(self.validation_labels, predicted_labels) # resulted labels VS desired ones
        confusionDisplay = ConfusionMatrixDisplay(confusionMatrix)
        confusionDisplay.plot()
        plt.title("Confusion matrix")
        plt.show()

    def seeScore(self):
        self.trainModel() # train the vectorizer
        train_vect, valid_vect, test_vect, test_valid_vect = self.getVectors() #collection of text --> matrix token counts(with diff proprieties)
        self.model.fit(train_vect, self.training_labels) # train the model
        predictions = self.model.predict(valid_vect)
        self.confusionMatrix(predictions)   # plot the matrix
        return self.model.score(valid_vect, self.validation_labels) # return the score


# if __name__ == "__main__":
#
#     tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
#     countVect = CountVectorizer(analyzer='char', ngram_range=(2, 6))
#
#     model = svm.SVC(C=1.0, kernel='linear')
#
#     option = int(input("1.SVM with tfidf \n2.SVM with CountVectorizer\n"))
#
#     svm = None
#     if(option == 1):
#         svm = SVM(model,tfidf)
#     else:
#         svm = SVM(model, countVect)
#     print(svm.seeScore())
