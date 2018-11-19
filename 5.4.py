import TMNIST
import time
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



c = [1,2,3,4,5,6,7]
gamma = [0.1, 0.2, 0.5, 1, 2, 5, 10]
parameters = {'gamma':gamma}
svc = svm.SVC(kernel='rbf', C=5)
clf = GridSearchCV(svc, parameters)
clf.fit(TMNIST.train_data, TMNIST.train_labels)
print clf.best_params_
#{'C': 5, 'gamma': 0.2}

print clf.cv_results_['mean_test_score']
#[ 0.9428  {0.9502}  0.9332  0.7498  0.3108  0.1834  0.14  ]

plt.scatter(gamma,clf.cv_results_['mean_test_score'])
plt.show()


degree = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 15, 20]
# coef0 = [1, 2, 3, 4, 5, 6, 7]
parameters = {'degree':degree}
svc = svm.SVC(coef0=1, kernel='poly')
clf = GridSearchCV(svc, parameters)
clf.fit(TMNIST.train_data, TMNIST.train_labels)
print clf.best_params_
# {'coef0': 1, 'degree': 9}
print clf.cv_results_['mean_test_score']
#[ 0.875   0.8934  0.9056  0.9166  0.9238  0.9264  0.9294  0.9324  0.9332  0.933   0.9314  0.927 ]
print clf.best_score_
#0.9332

plt.scatter(degree,clf.cv_results_['mean_test_score'])
plt.show()


pca = PCA()
pca.fit(TMNIST.train_data)

ratio = 0
n = 0
for r in pca.explained_variance_ratio_:
	ratio += r
	if(ratio > 0.95):
		break
	n += 1


#n=39
pca = PCA(n_components=n)
pca.fit(TMNIST.train_data)

train_d = pca.transform(TMNIST.train_data)
test_d = pca.transform(TMNIST.test_data)


start_time = time.time()
svc = svm.SVC(kernel='rbf', C=5, gamma=0.2).fit(train_d, TMNIST.train_labels)
train_time = time.time() - start_time

#print train_time

svc_results = svc.predict(test_d)

print (svc_results == TMNIST.test_labels).mean()