from sklearn.datasets import load_iris
iris = load_iris()
print(list(iris.target_names))

from sklearn import tree #imports sk learn models
# in order to make predictions we need to choose a classification model
classifier = tree.DecisionTreeClassifier() #assigns it to a variable (classifier) so we can work with it

classifier = classifier.fit(iris.data, iris.target)
#build the decision tree through which each new example will flow; can be built by feeding the training example and the target labels using the fit function

print(classifier.predict([[5.1,3.5,1.4,1.5]]))
#now we will make predictions

# the outer most set of square bracket is an array of our examples
# the inner most set of sq brackets is where we will put the values of features for a single example



#test the model
