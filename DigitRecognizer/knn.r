library(FNN);

train = read.csv('Data/train.csv',header=T);
test =read.csv('Data/test.csv',header=T);

lables = train[,1];
train = train[,-1];

results = knn(train,test,lables,k=10,algorithm="cover_tree");

#write(results,file="knn_benchmark.csv",ncolumns=1);
