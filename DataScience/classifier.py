# coding: utf-8
import csv;
import numpy as np;
from sklearn import svm,neighbors,cross_validation,ensemble;

def load_datasets(fname):
	return list(csv.reader(open(fname,'rb')));

if __name__ == '__main__':
	train_feature = np.array(load_datasets('Data/train.csv')
				,dtype=np.float);
	train_target = np.array(load_datasets('Data/trainLabels.csv'),dtype=np.int);

	test_feature = np.array(load_datasets('Data/test.csv'),dtype=np.float);
	"""
	svm = svm.SVC(C=1e-10,kernel='linear');
	svm.fit(train_feature,train_target);

	print svm.score(train_feature,train_target);
	"""
	#knn = neighbors.KNeighborsClassifier(weights='uniform');
	rfc = ensemble.RandomForestClassifier(n_estimators=10);
	#for i in xrange(1,21):
	#knn.fit(train_feature,train_target);
		#knn.n_neighbors = i;
		#rfc.n_estimators = i;
		#result = cross_validation.cross_val_score(rfc,train_feature,train_target,n_jobs=1,cv=10);
		#print "(%d)mean:%f sd:%f " % (i,np.mean(result),np.std(result));
	#print knn.score(train_feature,train_target);
	#result = knn.predict(train_feature);
	#print len(map(lambda x,y: x==y,result,train_target));

	rfc.fit(train_feature,train_target);
	result = rfc.predict(test_feature);
	#print result;
	out_f = file('result.csv','w');
	writer = csv.writer(out_f);
	writer.writerow(['Id','Solution']);
	for i,r in enumerate(result,start=1):
		writer.writerow([i,r]);
	out_f.close();


	
