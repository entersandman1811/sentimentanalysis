from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

data = load_files('/home/souradeep/txt_sentoken')

print data.data[0:10]


params = {
          "svc__C": [.01, .1, 1, 10, 100]}

clf = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True)),
                ("svc", LinearSVC())])

gs = GridSearchCV(clf, params, verbose=2, n_jobs=-1)
gs.fit(data.data, data.target)
print(gs.best_estimator_)
print(gs.best_score_)