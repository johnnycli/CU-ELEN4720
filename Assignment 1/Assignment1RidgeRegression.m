%Assignment1RidgeRegression.m
% Johnny C. Li jcl2222@columbia.edu
% This program takes the test data and perform a ridge regression with 
% various values for lambda. Plots regression reuslts against degrees of
% freedom and RMSE against lambda.  

%Load x data from file
%global x_train_data;
x_train_data=csvread('X_train.csv',0,0);

%Load y data from file
y_train_data=csvread('y_train.csv',0,0);
y_train_response=y_train_data(:,1);

%Creates function to find the Wrr
wrr = @(lambda,y,X) inv(lambda*eye(7)+X'*X)*X'*y;
df = @(lambda,X) trace(X*inv((X'*X+lambda*eye(7)))*X');

%Initialize empty matrix for data storage.
wrrmatrix = [];
dfmatrix = [];

%For loop to calculate df and wrr at 5000 lambda and write to matrix.
for lam= 0:5000
    xtt=wrr(lam,y_train_data,x_train_data);
    dft=df(lam,x_train_data);
    wrrmatrix = [wrrmatrix xtt];
    dfmatrix = [dfmatrix dft];
end

%Organize the results for plotting
combdata = [dfmatrix' wrrmatrix'];

%Plot out the result
figure
hold on
title('Ridge Regression on Training Set')
xlabel('df(\lambda)') 
plot(combdata(:,1),combdata(:,2));
plot(combdata(:,1),combdata(:,3));
plot(combdata(:,1),combdata(:,4));
plot(combdata(:,1),combdata(:,5));
plot(combdata(:,1),combdata(:,6));
plot(combdata(:,1),combdata(:,7));
plot(combdata(:,1),combdata(:,8));
legend({'x_1','x_2','x_3','x_4','x_5','x_6','x_7'},'Location','southwest')
hold off

%Load x test data from file
%global x_test_data;
x_test_data=csvread('X_test.csv',0,0);

%Load y test data from file
y_test_data=csvread('y_test.csv',0,0);

%Define varialbes for storage
summatrix = [];
RMSE = [];
%Outer for loop to calculate RSME for 5000 lambdas.
for lam= 0:50
    xtest=x_test_data*wrrmatrix(:,lam+1);
    %Difference between prediction and actual data
    summatrix=(x_test_data*wrrmatrix(:,lam+1)-y_test_data).^2;
    sumval = sum(summatrix,'all');
    %Calculate RSME
    RMSE = [RMSE sqrt(sumval/42)];
end

%Combine data for plotting purposes
RMSEdata = [(0:1:50)' RMSE'];

%Plot out the result
figure
hold on
title('RMSE for Ridge Regression')
xlabel('\lambda') 
plot(RMSEdata(:,1),RMSEdata(:,2));
hold off
