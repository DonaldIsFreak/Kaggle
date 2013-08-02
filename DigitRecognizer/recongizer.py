import csv,cPickle;
import numpy as np;
from sklearn import svm,datasets;
	
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
	svc = svm.SVC();
	svc.fit(train_feature,train_target);

	cPickle.dump(svc,open('svm.pkl','wb'));

	result = svc.predict(test_feature);
	cPickle.dump(result,open('result.pkl','wb'));

	print result;

	pass;



