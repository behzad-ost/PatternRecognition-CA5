from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas
from IPython.display import display, HTML

iris = datasets.load_iris()
parameters = {'decision_function_shape': ['ovr', 'ovo'] , 'degree':[1, 2, 3,4,5,6, 10]}
svc = svm.SVC(kernel='poly')
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
             
# d = pandas.DataFrame(clf.cv_results_)
print clf.best_params_