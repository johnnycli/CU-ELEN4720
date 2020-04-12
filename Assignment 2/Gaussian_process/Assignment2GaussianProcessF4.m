% Assignment2GaussianProcessF4.m
% Johnny C. Li jcl2222@columbia.edu
%
% This program takes the test data and perform a Gaussian process
% regression using only feature 4 with b=5 and sigsq=2.
%

%Load x data from file
x_train_data=csvread('X_train.csv',0,0);

%Load y data from file
y_train_data=csvread('y_train.csv',0,0);
y_train_response=y_train_data(:,1);

%Load x test data from file
x_test_data=csvread('X_test.csv',0,0);

%Load y test data from file
y_test_data=csvread('y_test.csv',0,0);

%Define the Kernel function with Euclidean norm
kern = @ (xi,xj,b) exp(-1/b*norm(xi-xj)^2);
%n-n Identity
Ikk = eye(size(x_train_data,1));

b=5;
sigsq=2;
%Predefine matrix for Kn
kmatrix=zeros(size(x_train_data,1),size(x_train_data,1));
%Calculate the values for Kn
%Calculate the values for Kij and populate matrix K.
        for i=1:size(x_train_data,1)
            for j=1:size(x_train_data,1)
                kmatrix(i,j)= kern(x_train_data(i,4),x_train_data(j,4),b);
            end
        end


muxres=zeros(1,size(x_train_data,1));
%Construct the kxDn row matrix.
for p=1:size(x_train_data,1)
    xtest=x_train_data(p,4);
    %Initialize and generate kxDn 
    kxDn=zeros(1,size(x_train_data,1));
            for i=1:size(x_train_data,1)
                kxDn(i)=kern(xtest,x_train_data(i,4),b);
            end
    muxres(p) = kxDn/(((sigsq)*Ikk+kmatrix))*y_train_data(:,1);
end

%Sort matrix for plotting
tmatrix=cat(2,x_train_data(:,4),muxres.');
[~,idx] = sort(tmatrix(:,1));
datamatrix = tmatrix(idx,:); 


figure()
hold on
scatter(x_train_data(:,4),y_train_data(:,1));
plot(datamatrix(:,1),datamatrix(:,2),'-r')

title('Gaussian Process with only Feature 4')
xlabel('x_4') 
ylabel('y') 
legend('x[4] data','Predictive mean')
hold off




