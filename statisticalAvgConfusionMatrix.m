function avgConfusion=statisticalAvgConfusionMatrix(numIterations,numNeighbours,percentage_training)

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);


nMat=[];
for j=1:numIterations

% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);

% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class3 test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);


% Note that feature 1 is in the columns,feature 16 is in the columns 

cTrain1=[train_samples_class1 ones(length(train_samples_class1),1)];
cTrain2=[train_samples_class2 2*ones(length(train_samples_class2),1)];
cTrain3=[train_samples_class3 3*ones(length(train_samples_class3),1)];

trainFeatures=[cTrain1;cTrain2;cTrain3];

cTest1=[test_samples_class1 ones(length(test_samples_class1),1)];
cTest2=[test_samples_class2 2*ones(length(test_samples_class2),1)];
cTest3=[test_samples_class3 3*ones(length(test_samples_class3),1)];

testFeatures=[cTest1;cTest2;cTest3];


predicted=[];
actual=[];


mdl=ClassificationKNN.fit(trainFeatures(:,1:num_features),trainFeatures(:,num_features+1),'NumNeighbors',numNeighbours);

predicted=[];
actual=[];
for i=1:length(testFeatures)
predicted =[predicted predict(mdl,testFeatures(i,1:num_features))];
actual=[actual testFeatures(i,num_features+1)];
end

confusionMatrix=confusionmat(predicted,actual);

normalMat=(1/(length(testFeatures)/3))*confusionMatrix
normalMat=transpose(normalMat)
nMat(:,:,j)=normalMat;
end

temp=[0 0 0;0 0 0;0 0 0];
for k=1:numIterations
 temp=temp+nMat(:,:,k) ;
end
avgConfusion=temp/numIterations;

end