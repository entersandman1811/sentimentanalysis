
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Import some data to play with
iris = datasets.load_iris()
X = iris.data

print X
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]



# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = MultinomialNB()


print y_train[:,1]
classifier.fit(X_train, y_train[:,1])


y_score = classifier.predict_proba(X_test)[:,1]
#Compute ROC curve and ROC area for each class
print y_score

fpr, tpr, _ = roc_curve(y_test[:, 1], y_score)
roc_auc = auc(fpr, tpr)




##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()