% testing function
clear all
clc
close all


load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

class1=y(1:7660,1:16);
class2=y(7660:7660*2,1:16);
class3=y(7660*2:7660*3,1:16);

percentage_training=70;
percentage_testing=30;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);

percentage_training=70;
percentage_testing=30;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


percentage_training=70;
percentage_testing=30;
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


mdl=ClassificationKNN.fit(trainFeatures(:,1:16),trainFeatures(:,17),'NumNeighbors',3);

predicted=[]
actual=[]
for i=1:length(testFeatures)
predicted =[predicted predict(mdl,testFeatures(i,1:16))];
actual=[actual testFeatures(i,17)];
end

confusionMatrix=confusionmat(predicted,actual);

normalMat=(1/(length(testFeatures)/3))*confusionMatrix


% normalMat =
% 
%     0.9896    0.0044    0.0152
%     0.0004    0.8903    0.0753
%     0.0100    0.1053    0.9095

