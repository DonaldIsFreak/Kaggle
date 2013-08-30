import cPickle,csv;
import MyDataSet as myData;
import numpy as np;
from sklearn.ensemble import RandomForestClassifier;

def dump_file(result,fname='result.pkl'):
	cPickle.dump(result,open(fname,'wb'));

def load_file(result,fname='result.pkl'):
	return cPickle.load(open(fname,'rb'));

if __name__ == '__main__':
	digits = myData.load_digits();
	rfc = RandomForestClassifier();
	rfc.fit(digits.feature,digits.target);
	print rfc.score(digits.feature,digits.target);
	
	result = rfc.predict(digits.test);
	out_f = file('result.csv','w');
	writer = csv.writer(out_f);
	writer.writerow(['Id','Solution']);
	for i,r in enumerate(result,start=1):
		writer.writerow([i,r]);
	out_f.close();
	#dump_file(rfc,'rfc.pkl');
	#dump_file(result,'result3.pkl');
	pass;
