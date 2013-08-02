import csv,cPickle;
import numpy as np;
from sklearn import svm,datasets,neighbors;
	
if __name__ == '__main__':
	r = csv.reader(open('Data/train.csv','r'));
	mat = np.array(list(r)[1:]);

	train_target = mat[:,0];
	train_feature = mat[:,1:];

	r = csv.reader(open('Data/test.csv','r'));
	test_feature = np.array(list(r)[1:]);

	"""
	# Test Script 

	digits = datasets.load_digits();
	train_feature = digits.data;
	train_target = digits.target;

	test_feature = digits.data[:20];
	"""

	knn = neighbors.KNeighborsClassifier(n_neighbors=10);
	knn.fit(train_feature,train_target);

	cPickle.dump(knn,open('knn.pkl','wb'));

	result = knn.predict(test_feature);
	cPickle.dump(result,open('result2.pkl','wb'));

	print result;

	pass;



