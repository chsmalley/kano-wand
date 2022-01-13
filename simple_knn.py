import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


#custom metric
def DTW(a, b): 
    print(f"a: {a}")  
    print(f"b: {b}")  
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),
                                        b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    print(f"cumdist: {cumdist[an, bn]}")
    return cumdist[an, bn]

def main():
    #toy dataset 
    X = np.random.random((100,10))
    y = np.random.randint(0,2, (100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(f"X_train: {X_train}")
    print(f"X_test: {X_test}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")
    #train
    parameters = {'n_neighbors':[2, 4, 8]}
    clf = GridSearchCV(KNeighborsClassifier(metric=DTW),
                       parameters,
                       cv=3,
                       verbose=1)
    clf.fit(X_train, y_train)
    #evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()