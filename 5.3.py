import TMNIST
from sklearn import svm
import prettytable
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

# parameters = {'C':[1,2,3,4,5,6,7], 'degree':[1,2,3,4,5,6,7]}
# svc = svm.SVC(kernel='poly')
# clf = GridSearchCV(svc, parameters)
# clf.fit(TMNIST.train_data, TMNIST.train_labels)
# print clf.best_params_
#{'C': 7, 'degree': 1}
# print clf.cv_results_['mean_test_score']


start_time = time.time()
ovr = OneVsRestClassifier(svm.SVC(kernel='poly', degree=1)).fit(TMNIST.train_data, TMNIST.train_labels)
train_time = time.time() - start_time
print train_time

start_time = time.time()
ovr_res = ovr.predict(TMNIST.test_data)
test_time = time.time() - start_time
print test_time

start_time = time.time()
ovo = OneVsOneClassifier(svm.SVC(kernel='poly', degree=1)).fit(TMNIST.train_data, TMNIST.train_labels)
train_time = time.time() - start_time
print train_time

start_time = time.time()
ovo_res = ovo.predict(TMNIST.test_data)
test_time = time.time() - start_time
print test_time

print (ovr_res == TMNIST.test_labels).mean()
print (ovo_res == TMNIST.test_labels).mean()

#ovo and ovr have same results with c=7 and sklearn decision_function_shape='ovo/ovr'!!!


#5.3.2
# x = [1,2,3,4,5]
# parameters = {'degree':x}
# svc = svm.SVC(kernel='poly')
# clf = GridSearchCV(svc, parameters)
# clf.fit(TMNIST.train_data, TMNIST.train_labels)
# print clf.best_params_
# print clf.cv_results_['mean_test_score']
# print clf.best_score_

# plt.scatter(x,clf.cv_results_['mean_test_score'])
# plt.show()


#5.3.4
# x = [0.1,0.2,0.5,1,5,10,15,20,30,50]
# parameters = {'C':x}
# svc = svm.LinearSVC()
# clf = GridSearchCV(svc, parameters)
# clf.fit(TMNIST.train_data, TMNIST.train_labels)
# print clf.best_params_
# 15
# print clf.cv_results_['mean_test_score']
# print clf.best_score_

# plt.scatter(x,clf.cv_results_['mean_test_score'])
# plt.show()