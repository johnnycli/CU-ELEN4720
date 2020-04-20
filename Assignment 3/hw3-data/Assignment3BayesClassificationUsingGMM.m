% Assignment3BayesClassificationUsingGMM.m
% Johnny C. Li jcl2222@columbia.edu
% Using the best run for each class after 30 iterations, predict the
% testing data using a Bayes classifier.
%
% Assume complimented code Assignment3EMforVariableGMMExtraction.m is ran
% first.
%

%% Load Data
%Load x data from file
x_test_data=csvread('Prob2_Xtest.csv',0,0);

%Load y data from file
y_test_data=csvread('Prob2_ytest.csv',0,0);


%% Bayes Classification
%Extract number of GMM from data.
NGMMP=size(mu_0_max,2);

%Only have two classes
arg_max = zeros(1,2);

%Allocate space for prediction
y_prediction = zeros(size(y_test_data,1),1);

p_x_mu_sig = @(x,mu,sig) 1/((2*pi)^(10/2)*det(sig).^(1/2))*exp(-1/2*transpose(x-mu)/sig*(x-mu));

%Find max Gaussian
[MaxG_1,G1] = max(pi_k_1_max);
[MaxG_0,G0] = max(pi_k_0_max);

%Confusion matrix (Prediction 0 1/Row)x(Actual 0 1/Column)
CM = zeros(2,2);
prob0 = 0;
prob1 = 0;

for i=1:size(y_test_data,1)
    %prob0=pi_k_0_max(G0)*p_x_mu_sig(x_test_data(i,:).',mu_0_max(:,G0),sig_0_max{G0});
    %prob1=pi_k_1_max(G1)*p_x_mu_sig(x_test_data(i,:).',mu_1_max(:,G1),sig_1_max{G1});
    
    for k=1:NGMMP
    prob0 = prob0 + pi_k_0_max(k)*p_x_mu_sig(x_test_data(i,:).',mu_0_max(:,k),sig_0_max{k});
    prob1 = prob1 + pi_k_1_max(k)*p_x_mu_sig(x_test_data(i,:).',mu_1_max(:,k),sig_1_max{k});
    end
    
    if(prob0>prob1)
        y_prediction(i)=0;
    else
        y_prediction(i)=1;
    end
    
    if(y_test_data(i)==1)
        if(y_prediction(i)==1)
            CM(2,2)=CM(2,2)+1;
        else
            CM(2,1)=CM(2,1)+1;
        end
    else
        if(y_prediction(i)==1)
            CM(1,2)=CM(1,2)+1;
        else
            CM(1,1)=CM(1,1)+1;
        end
    end
end

accuracy = (CM(1,1)+CM(2,2))/sum(CM,'all');

