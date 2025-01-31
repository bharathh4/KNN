clc
clear all
close all

profile on

numIterations=10;
numNeighbours=1;
percentage_training=90;
y=statisticalAvgConfusionMatrix(numIterations,numNeighbours,percentage_training)


profile viewer
p = profile('info');
profsave(p,'profile_results')

profile off

%%
%% iter=10, percentage_training=90;Change Rest

% numIterations=10;
% numNeighbours=1;
% percentage_training=90;
% Time=240.67s
% y =
% 
%     0.9906    0.0013    0.0081
%     0.0020    0.9108    0.0872
%     0.0093    0.0764    0.9144


% numIterations=10;
% numNeighbours=3;
% percentage_training=90;
% Time=243s
% y =
% 
%     0.9894    0.0021    0.0085
%     0.0035    0.9051    0.0914
%     0.0124    0.0749    0.9127

% numIterations=10;
% numNeighbours=5;
% percentage_training=90;
% Time =244.86s
% y =
% 
%     0.9884    0.0013    0.0103
%     0.0048    0.8948    0.1004
%     0.0124    0.0710    0.9166



%
% numIterations=10;
% numNeighbours=9;
% percentage_training=90;
% Time=247.34s
% y =
% 
%     0.9855    0.0017    0.0128
%     0.0061    0.8907    0.1031
%     0.0138    0.0798    0.9064

% numIterations=10;
% numNeighbours=72;
% percentage_training=90;
% Time=246.39s
% y =
% 
%     0.9565    0.0043    0.0392
%     0.0176    0.7909    0.1915
%     0.0316    0.0868    0.8816


%%
display('Some of these observations show that more neighbours are degrade error rates')
display('Also the number of neighbours does not greatly influence the search times.Infact it does not')


%% iter=1, percentage_training=90;Change Rest

% numIterations=1;
% numNeighbours=3;
% percentage_training=90;
% Time=27.91s
% y =
% 
%     0.9896    0.0026    0.0078
%     0.0052    0.9308    0.0640
%     0.0078    0.0744    0.9178



%%

%% iter=1, percentage_training=70;Change Rest
% numIterations=1;
% numNeighbours=3;
% percentage_training=70;
% Time = 71.10 s
% y =
% 
%     0.9891    0.0022    0.0087
%     0.0065    0.9008    0.0927
%     0.0178    0.0722    0.9099


% numIterations=1;
% numNeighbours=5;
% percentage_training=70;
% Time =72.83
% y =
% 
%     0.9848    0.0017    0.0135
%     0.0044    0.8947    0.1010
%     0.0161    0.0792    0.9047

% numIterations=1;
% numNeighbours=9;
% percentage_training=70;
% Time=72.5
% y =
% 
%     0.9813    0.0026    0.0161
%     0.0083    0.8773    0.1144
%     0.0191    0.0779    0.9030

% numIterations=1;
% numNeighbours=18;
% percentage_training=70;
% Time=72.19
% y =
% 
%     0.9778    0.0052    0.0170
%     0.0122    0.8507    0.1371
%     0.0239    0.0779    0.8982

% numIterations=1;
% numNeighbours=36;
% percentage_training=70;
% Time=71.85s
% y =
% 
%     0.9678    0.0057    0.0265
%     0.0139    0.8403    0.1458
%     0.0270    0.0927    0.8803

% numIterations=1;
% numNeighbours=72;
% percentage_training=70;
% Time=71.768
% y =
% 
%     0.9569    0.0048    0.0383
%     0.0200    0.7859    0.1941
%     0.0366    0.0857    0.8777

%%
%% Percentage_train=50,Iter=1,Change Rest
% numIterations=1;
% numNeighbours=3;
% percentage_training=50;
% Time=105.02s
% y =
% 
%     0.9851    0.0018    0.0131
%     0.0084    0.8809    0.1107
%     0.0133    0.0807    0.9060

% numIterations=1;
% numNeighbours=9;
% percentage_training=50;
% Time=104.33s
% y =
% 
%     0.9775    0.0034    0.0191
%     0.0125    0.8572    0.1303
%     0.0222    0.0799    0.8979

% numIterations=1;
% numNeighbours=36;
% percentage_training=50;
% Time = 105.54s
% y =
% 
%     0.9564    0.0052    0.0384
%     0.0201    0.8107    0.1692
%     0.0274    0.0849    0.8877

% numIterations=1;
% numNeighbours=72;
% percentage_training=50;
% Time=105.369s
% y =
% 
%     0.9392    0.0034    0.0574
%     0.0258    0.7593    0.2149
%     0.0392    0.0828    0.8781

%%
%% Percentage_train=20,Iter=1,Change Rest
% numIterations=1;
% numNeighbours=3;
% percentage_training=20;
% Time=131.5s
% y =
% 
%     0.9735    0.0042    0.0222
%     0.0134    0.8402    0.1465
%     0.0286    0.0925    0.8790

% numIterations=1;
% numNeighbours=5;
% percentage_training=20;
% T=132.83s
% y =
% 
%     0.9676    0.0034    0.0289
%     0.0148    0.8444    0.1408
%     0.0346    0.1072    0.8583

% numIterations=1;
% numNeighbours=9;
% percentage_training=20;
% Time =131.62
% y =
% 
%     0.9612    0.0055    0.0331
%     0.0163    0.8206    0.1632
%     0.0302    0.1025    0.8674

% numIterations=1;
% numNeighbours=18;
% percentage_training=20;
% Time = 132.78s
% y =
% 
%     0.9609    0.0036    0.0354
%     0.0214    0.7914    0.1873
%     0.0403    0.0990    0.8607

% numIterations=1;
% numNeighbours=36;
% percentage_training=20;
% Time = 132.715
% y =
% 
%     0.9232    0.0150    0.0617
%     0.0228    0.7626    0.2146
%     0.0364    0.1057    0.8579

% numIterations=1;
% numNeighbours=72;
% percentage_training=20;
% Time=134.59s
% y =
% 
%     0.8850    0.0176    0.0972
%     0.0269    0.6959    0.2772
%     0.0470    0.0940    0.8591

%%