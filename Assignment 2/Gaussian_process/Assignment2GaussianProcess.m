% Assignment2GaussianProcess.m
% Johnny C. Li jcl2222@columbia.edu
%
% This program takes the test data and perform a Gaussian process
% regression. The code does not include any sort of optimization or
% calculations for hyperparameters b and sigma squared; the values are
% taken from part (b) of the assignment sheet.
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

%Create array for constants
bval=(5:2:15);
sigsqval=(0.1:0.1:1);

%Define the Kernel function with Euclidean norm
kern = @ (xi,xj,b) exp(-1/b*norm(xi-xj)^2);
%n-n Identity
Ikk = eye(size(x_train_data,1));
%Predefine the RMSE result matrix
RMSEmatrix=zeros(6,10);
%Predefine matrix for Kn
kmatrix=zeros(size(x_train_data,1),size(x_train_data,1));

%Start to run through the 60 combinations of hyperparameters.
for bi=1:6
    for si=1:10
        %Calculate the values for Kij and populate matrix K.
        for i=1:size(x_train_data,1)
            for j=1:size(x_train_data,1)
                kmatrix(i,j)= kern(x_train_data(i,:),x_train_data(j,:),bval(bi));
            end
        end
        %muxres is the result for 1 out of 60 runs.
        muxres=zeros(1,size(x_test_data,1));
        %Construct the kxDn row matrix.
        for p=1:size(x_test_data,1)
            xtest=x_test_data(p,:);
            %Initialize and generate kxDn 
            kxDn=zeros(1,size(x_train_data,1));
            for i=1:size(x_train_data,1)
                kxDn(i)=kern(xtest,x_train_data(i,:),bval(bi));
            end
            muxres(p) = kxDn*inv(((sigsqval(si))*Ikk+kmatrix))*y_train_data(:,1);
        end
        RMSEmatrix(bi,si)=sqrt(1/size(y_test_data,1)*sum((y_test_data-muxres.').^2,'all'));
    end
end



