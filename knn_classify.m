function predicted=knn_classify(numNeighbours,percentage_training,feature)
% This function classifies a single feature vector into class 1 or class 2
% or class 3.
% Its inputs are num_neighbours,percentage_training and the feature vector
% that needs classification


load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

class1=y(1:7660,1:16);
class2=y(7660:7660*2,1:16);
class3=y(7660*2:7660*3,1:16);

% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class1, test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);

% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class2, test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


% percentage_training=70;
percentage_testing=100-percentage_training;
[train_samples_class3, test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);


% Note that feature 1 is in the columns,feature 16 is in the columns 

cTrain1=[train_samples_class1 ones(length(train_samples_class1),1)];
cTrain2=[train_samples_class2 2*ones(length(train_samples_class2),1)];
cTrain3=[train_samples_class3 3*ones(length(train_samples_class3),1)];

trainFeatures=[cTrain1;cTrain2;cTrain3];

mdl=ClassificationKNN.fit(trainFeatures(:,1:16),trainFeatures(:,17),'NumNeighbors',numNeighbours);

predicted =predict(mdl,feature);

end
