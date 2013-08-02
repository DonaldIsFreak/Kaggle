
import numpy as np;
from sklearn.ensemble import RandomForestClassifier;

if __name__ == '__main__':
	dataSet = np.genfromtxt('./Data/train.csv',dtype='f8',delimiter=',')[1:];
	test = np.genfromtxt('./Data/test.csv',dtype='f8',delimiter=',')[1:];
	target = dataSet[:,0];
	feature = dataSet[:,1:];
	rf = RandomForestClassifier(n_estimators=100);
	rf.fit(feature,target);

	for i in rf.predict_proba(test):
		print i[1];
	pass;
