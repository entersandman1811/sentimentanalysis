from sklearn.feature_extraction.text import CountVectorizer
import os
import glob
from sklearn import grid_search
import time
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

path = "/home/souradeep/Downloads/aclImdb_full"

def grab_data(path,y):
    sentences = []


    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    types = ("*.txt","*.txt~")
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    for ff in files_grabbed:
        with open(ff, 'r') as f:
            sentences.append(f.read().strip())
            y.append(1)
    os.chdir('%s/neg/' % path)
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    for ff in files_grabbed:
        with open(ff, 'r') as f:
            sentences.append(f.read().strip())
            y.append(0)
    os.chdir(currdir)
    count = 0
    # with open("sentences.txt", "w+1") as the_file:
    #     for item in sentences:
    #         the_file.write("%s\n" % item)
    #         count += 1
    #         if count == 100:
    #             break
    return sentences

def main():

    train = True

    print "Train data? yes or no"

    response = raw_input()

    if response == 'no' or response == 'n':   train =False
    print "Retrieving data..."

    y=[]
    sentences = grab_data(path, y)
    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000)
    all_data_features = vectorizer.fit_transform(sentences)
    all_data_features = all_data_features.toarray()
    train_data_features, test_data_features, y_train, y_test = cross_validation.train_test_split(
        all_data_features, y, test_size=0.2, random_state=1)

    if train == True:
        start_time = time.time()
        # count =0
        # with open("bow_word_vectors.txt","w+1") as the_file:
        #     for item in train_data_features:
        #         the_file.write("\n" )
        #         for value in item:
        #             the_file.write("%d " %value)
        #         count +=1
        #         if count == 100:
        #             break
        svr = MultinomialNB()
        parameters = { 'alpha': np.logspace(-2, 10, 13)}
        clf_bow = grid_search.GridSearchCV(svr, parameters,n_jobs=-1)
        print "Starting training..."

        clf_bow.fit(train_data_features, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (clf_bow.best_params_, clf_bow.best_score_))
        print("Time to train the model: %s seconds " % (time.time() - start_time))

        _ = joblib.dump(clf_bow, "clf_bow.pkl", compress=9)


    start_time = time.time()
    #y_test = []

    clf_bow = joblib.load("clf_bow.pkl")

    #print "Retrieving test data..."

    # test_data = grab_data(os.path.join(path, 'test'), y_test)
    # test_data_features = vectorizer.fit_transform(test_data)
    # test_data_features = test_data_features.toarray()

    print "Predicting sentiments..."

    y_hat = clf_bow.predict_proba(test_data_features)[:,1]#for roc
    y_hat = clf_bow.predict(test_data_features)
    target_names = ['positive', 'negative']

    print(classification_report(y_test, y_hat, target_names=target_names))

    print("Time to predict sentiments: %s secs " % ((time.time() - start_time)))

    acc = accuracy_score(y_test,y_hat)
    print ("The accuracy is %s " %acc)

    # fpr, tpr, _ = roc_curve(y_test, y_hat)
    # roc_auc = auc(fpr, tpr)

    ##############################################################################
    # Plot of a ROC curve for a specific class

    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    cm = confusion_matrix(y_test, y_hat)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm_normalized)
    plt.show()
if __name__=="__main__":
    main()