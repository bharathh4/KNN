% This is just a data file that makes an observation

%%
clc
clear all

c=(7760)./100

num_train_samples=c*[90 70 50 20]
errors=[0.956 0.7909 0.881; 0.9569 0.7859 0.8777; 0.9392 0.7593 0.8791; 0.8850 0.6959 0.8591];
bar(num_train_samples,errors)
grid

xlabel('Number of training samples/search space')
ylabel('Classification success rate/diagonal elements of confusion matrix')
title('Error rates for 72 neighbours.A large number is chosen to illustrate')

%% Do this for 3 neighbours as well

%%

Display('This is enough to conclude that more training data/search space gives better results')

% numIterations=10;% Forget this
% numNeighbours=72;
% percentage_training=90;
% Time=246.39s
% y =
% 
%     0.9565    0.0043    0.0392
%     0.0176    0.7909    0.1915
%     0.0316    0.0868    0.8816

% numIterations=1;
% numNeighbours=72;
% percentage_training=70;
% Time=71.768
% y =
% 
%     0.9569    0.0048    0.0383
%     0.0200    0.7859    0.1941
%     0.0366    0.0857    0.8777

% numIterations=1;
% numNeighbours=72;
% percentage_training=70;
% y =
% 
%     0.9392    0.0034    0.0574
%     0.0258    0.7593    0.2149
%     0.0392    0.0828    0.8781

% numIterations=1;
% numNeighbours=72;
% percentage_training=20;
% Time=134.59s
% y =
% 
%     0.8850    0.0176    0.0972
%     0.0269    0.6959    0.2772
%     0.0470    0.0940    0.8591

