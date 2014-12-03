

clear all
clc
close all

%Class 1
class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
Class1_X=[class_1_x1 ;class_1_x2 ];
Class1_X=transpose(Class1_X);%Each feature in a column


%Class 2
class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
Class2_X=[class_2_x1 ;class_2_x2 ];
Class2_X=transpose(Class2_X);%Each feature in a column


%Class 3
class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
Class3_X=[class_3_x1 ;class_3_x2 ];
Class3_X=transpose(Class3_X);%Each feature in a column


X= [Class1_X ;Class2_X ;Class3_X];
Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];

% rloss=[];
% for i=1:1:100
% 
% mdl=ClassificationKNN.fit(X,Y,'NumNeighbors',i);
% 
% rloss=[rloss resubLoss(mdl)];
% 
% end
% 
% plot(rloss)

mdl=ClassificationKNN.fit(X,Y,'NumNeighbors',6);

predicted=[]
actual=[]
for i=1:length(X)
predicted =[predicted predict(mdl,X(i,:))];
actual=[actual Y(i)];
end


confusionMatrix=confusionmat(predicted,actual);
%normalMat=(1/length(test_samples_class1))*confusionMatrix
% confusionMatrix =
% 
%         1932          84          26
%           53        1908          15
%           15           8        1959


% normalMat=(1/2000)*confusionMatrix
% normalMat =
% 
%     0.9660    0.0420    0.0130
%     0.0265    0.9540    0.0075
%     0.0075    0.0040    0.9795



