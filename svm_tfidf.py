from sklearn.feature_extraction.text import TfidfVectorizer
import os
import glob
from sklearn import svm
import time
from sklearn.externals import joblib
from sklearn.metrics import classification_report

path = "/home/souradeep/Downloads/aclImdb"

def grab_data(path,y):
    sentences = []


    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
            y.append(1)
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
            y.append(0)
    os.chdir(currdir)

    return sentences

def main():

    train = True

    print "Train data? yes or no"

    response = raw_input()

    if response == 'no' or response == 'n':   train =False

    vectorizer = TfidfVectorizer(min_df=1)

    if train == True:

        start_time = time.time()
        y_train = []
        print "Retrieving training data..."
        sentences = grab_data(os.path.join(path, 'train'), y_train)


        train_data_features = vectorizer.fit_transform(sentences)
        train_data_features = train_data_features.toarray()

        clf_tfidf = svm.SVC()

        print "Strating training..."

        clf_tfidf.fit(train_data_features, y_train)

        print("Time to train the model: %s seconds " % (time.time() - start_time))

        _ = joblib.dump(clf_tfidf, "clf.pkl", compress=9)


    start_time = time.time()
    y_test = []

    clf = joblib.load("clf.pkl")

    print "Retrieving test data..."

    test_data = grab_data(os.path.join(path, 'test'), y_test)
    test_data_features = vectorizer.fit_transform(test_data)
    test_data_features = test_data_features.toarray()

    print "Predicting sentiments..."

    y_hat = clf.predict(test_data_features)
    target_names = ['positive', 'negative']

    print(classification_report(y_test, y_hat, target_names=target_names))

    print("Time to predict sentiments: %s secs " % ((time.time() - start_time)))

if __name__=="__main__":
    main()