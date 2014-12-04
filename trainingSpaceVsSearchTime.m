% This file plots the avg search times vs the training space
clc
clear all

c=(7760)./100

num_test_samples=c*[10 30 50 80]
time=[24.5 72 105 132]

% hold on

num_train_samples=c*[90 70 50 20]


normalTimes=time./num_test_samples % Time taken to execute per sample

stem(num_train_samples,normalTimes)
xlabel('Number of training samples/search space')
ylabel('Avg time to classify a point in seconds')
grid




