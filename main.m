% clear all
close all
clc

X = (linspace(0,10))';
Y = sin(X);
% trainData_3 = [X(1:70),Y2(1:70)];
% testData_3 = [X(71:100),Y2(71:100)];
% CMAC_map = create_2(X,35);
% [CMAC_map_3,iteration_3,error_3,t_3] = train_2(CMAC_map,trainData_3,1);
% accu_3 = test_2(CMAC_map_3,testData_3);
% iteration = zeros(2,34);
% iter = zeros(2,34);
% accu = zeros(2,34);
% acc = zeros(2,34);
% t = zeros(2,34);
% T = zeros(2,34);
% for j=1:100
    I = randperm(100);
    trainData = [X(I(1:70)),Y(I(1:70))];
    testData = [X(I(71:100)),Y(I(71:100))];
    for i=1:34
        CMAC_map = create(X,35,i);
        figure
        plot(X,Y);
        hold on
        [map,iter(1,i),~,T(1,i)] = train(CMAC_map,trainData,0,0);
        acc(1,i) = test(map,testData,0);
        hold off
        legend('Main Function','Test Output');
        title(['Overlap = ' num2str(i)]);
        figure
        plot(X,Y);
        hold on
        [map,iter(2,i),~,T(2,i)] = train(CMAC_map,trainData,0,1);
        acc(2,i) = test(map,testData,1);
        hold off
        legend('Original Curve','Testing Data');
        title(['numCell = ' num2str(i)]);
    end
%     iteration = iteration + iter;
%     accu = accu + acc;
%     t = t + T;
% end
% iteration = iteration/j;
% accu = accu/j;
% t = t/j;
