%% Quick Test : Classify a feature 

clc
clear all
close all

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);

%Specify number of neighbours
numNeighbours=3;
%Specify amount of training data/search space
percentage_training=70;

%Switch on the profiler for timing information
%profile on

% Plug a feature vector from class 1
label=knn_classify(numNeighbours,percentage_training,class1(200,1:16)) % Should give 1 when we plug the 200th observation of class 1

% % Plug a feature vector from class 2
% label=knn_classify(numNeighbours,percentage_training,class2(200,1:16)) % Should give 2 when we plug the 200th observation of class 2
% 
% % Plug a feature vector from class 3
% label=knn_classify(numNeighbours,percentage_training,class3(200,1:16)) % Should give 3 when we plug the 200th observation of class 3

% Check profiler for timing information of the function
% profile viewer
% p = profile('info');
% profsave(p,'profile_results')
% profile off
%%

%% Calculate average confusion matrix.For a quick test set numIterations=2 and numNeighbours=3 

%Set number of times to re-randomize and calculate confusion matrix and
% average. Beware large values can take a long time to compute.
numIterations=1; 

% Set number of neighbours for majority vote
numNeighbours=3;
percentage_training=95;
% The average confusion matrix
avgConfusion=statisticalAvgConfusionMatrix(numIterations,numNeighbours,percentage_training)

%%

%% Compute confusion matrix for one pass with number of neighbours being 3 and training data percent being 70.Helps visualize the flow.
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

predicted=[];
actual=[];
for i=1:length(testFeatures)
predicted =[predicted predict(mdl,testFeatures(i,1:16))];
actual=[actual testFeatures(i,17)];
end

confusionMatrix=confusionmat(predicted,actual);

normalMat=(1/(length(testFeatures)/3))*confusionMatrix
normalMat=transpose(normalMat)
%%


