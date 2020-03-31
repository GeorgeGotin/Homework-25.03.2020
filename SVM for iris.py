from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
X = data['data']
y = data['target']

X = X[:list(y).index(2)]
y = y[:list(y).index(2)]

model = PCA(n_components=2)

model.fit(X)
X = model.transform(X) 

svc = LinearSVC()
svc.fit(X,y)
p = svc.coef_[0]


'''
...was:
X0 = X[0:list(y).index(1)]
X1 = X[list(y).index(1):len(list(y))]

plt.plot([i[0] for i in X0],[i[1] for i in X0],'o')
plt.plot([i[0] for i in X1],[i[1] for i in X1],'o')
plt.show()

...became:
'''

plt.plot([i[0] for i in X[0:list(y).index(1)]],[i[1] for i in X[0:list(y).index(1)]],'o')
plt.plot([i[0] for i in X[list(y).index(1):len(list(y))]],[i[1] for i in X[list(y).index(1):len(list(y))]],'o')
plt.plot([i for i in range(-5,5)],[i*p[0]+p[1] for i in range(-5,5)])
plt.show()

