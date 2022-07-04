from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from Model import Model

tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
countVect = CountVectorizer(analyzer='char', ngram_range=(2, 6))

optionModel = int(input("1.Naive Bayes\n2.Neural Network\n3.SVM\n"))
model = None
repr = None
if (optionModel == 1):
    model = MultinomialNB()
    option = int(input("1.NB with tfidf \n2.NB with CountVectorizer\n"))
    if(option == 1):
        repr = tfidf
    else:
        repr = countVect
elif (optionModel == 2):
    model = MLPClassifier(learning_rate='adaptive',early_stopping=True, hidden_layer_sizes =(64,16), batch_size=1024, verbose = True)
    option = int(input("1.NN with tfidf \n2.NN with CountVectorizer\n"))
    if (option == 1):
        repr = tfidf
    else:
        repr = countVect
elif (optionModel == 3):
    model = svm.SVC(C=1.0, kernel='linear', verbose = True)
    option = int(input("1.SVM with tfidf \n2.SVM with CountVectorizer\n"))
    if (option == 1):
        repr = tfidf
    else:
        repr = countVect

modelChose = Model(model, repr)
print(modelChose.seeScore())

