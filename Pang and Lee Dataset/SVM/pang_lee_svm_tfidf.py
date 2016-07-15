from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
import glob
from sklearn import svm , grid_search
import time
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import cross_validation
from sklearn.svm import LinearSVC

path = "/home/souradeep/txt_sentoken"

def grab_data(path,y):
    sentences = []


    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.read().strip())
            y.append(1)
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.read().strip())
            y.append(0)
    os.chdir(currdir)

    return sentences

def main():

    train = True

    print "Train data? yes or no"

    response = raw_input()

    if response == 'no' or response == 'n':   train =False

    vectorizer = TfidfVectorizer(analyzer = "word",
                             )

    if train == True:

        start_time = time.time()
        y = []
        print "Retrieving training data..."
        sentences = grab_data(path, y)


        all_data_features = vectorizer.fit_transform(sentences)
        all_data_features = all_data_features.toarray()
        train_data_features, test_data_features, y_train, y_test = cross_validation.train_test_split(
            all_data_features, y, test_size=0.2, random_state=1)
        # count =0
        # with open("bow_word_vectors.txt","w+1") as the_file:
        #     for item in train_data_features:
        #         the_file.write("\n" )
        #         for value in item:
        #             the_file.write("%d " %value)
        #         count +=1
        #         if count == 100:
        #             break
        svr = LinearSVC()
        parameters = { 'C': np.logspace(-2, 10, 13)}
        clf_bow = grid_search.GridSearchCV(svr, parameters,n_jobs=-1)
        print "Strating training..."

        clf_bow.fit(train_data_features, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (clf_bow.best_params_, clf_bow.best_score_))
        print("Time to train the model: %s seconds " % (time.time() - start_time))

        _ = joblib.dump(clf_bow, "clf_bow.pkl", compress=9)


    start_time = time.time()
    #y_test = []

    clf_bow = joblib.load("clf_bow.pkl")

    print "Retrieving test data..."

    # test_data = grab_data(os.path.join(path, 'test'), y_test)
    # test_data_features = vectorizer.fit_transform(test_data)
    # test_data_features = test_data_features.toarray()

    print "Predicting sentiments..."

    y_hat = clf_bow.predict(test_data_features)
    target_names = ['positive', 'negative']

    print(classification_report(y_test, y_hat, target_names=target_names))

    print("Time to predict sentiments: %s secs " % ((time.time() - start_time)))

    acc = accuracy_score(y_test,y_hat)
    print ("The accuracy is %s " %acc)

if __name__=="__main__":
    main()