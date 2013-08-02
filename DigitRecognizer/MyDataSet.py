import csv;
import numpy as np;
class Bunch(dict):
	def __init__(self,**kwargs):
		dict.__init__(self,kwargs);
		self.__dict__ = self;


def load_digits(trainFileName='Data/train.csv',testFileName='Data/test.csv'):
	train = csv.reader(open(trainFileName,'r'));
	test = csv.reader(open(testFileName,'r'));
	train_data = np.array(list(train)[1:]);
	target = train_data[:,0];
	feature = train_data[:,1:];
	test_data = np.array(list(test)[1:]);

	return Bunch(train_data = train_data,feature=feature,target=target,test=test_data);
	
	
