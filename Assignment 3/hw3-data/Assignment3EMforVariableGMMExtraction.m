% Assignment3EMforVariableGMMExtraction.m
% Johnny C. Li jcl2222@columbia.edu
% Implement the EM algorithm for the Gaussian mixture model, with the 
% purpose of using it in a Bayes classifier. Extract optimal parameter for
% Bayes clasifier. 
%

%% Load Data
%Load x data from file
x_train_data=csvread('Prob2_Xtrain.csv',0,0);

%0-1631 1, 1632-4140 0
x_data = cell(1,2);
x_data{1}=x_train_data(1:1631,:);
x_data{2}=x_train_data(1632:4140,:);

%Load y data from file
y_train_data=csvread('Prob2_ytrain.csv',0,0);
y_train_response=y_train_data(:,1);

%% Control initialization
iteration = 30;
rounds = 10;
NGMM = 3;
L_it_lst=zeros(rounds,iteration);
%Defined my own function for calculating the PDF
p_xmu_sig = @(x,mu,sigdet,sigi) 1/((2*pi)^(10/2)*sigdet.^(1/2))*exp(-1/2*transpose(x-mu)*sigi*(x-mu));

%Pre-allocate space for data extraction
pi_k_1_data = cell(1,rounds);
mu_1_data=cell(1,rounds);
sig_1_data=cell(1,rounds);
L_it_1=zeros(1,rounds);
pi_k_0_data = cell(1,rounds);
mu_0_data=cell(1,rounds);
sig_0_data=cell(1,rounds);
L_it_0=zeros(1,rounds);

for d=1:2
x_train_data = x_data{d};

    
%% Multiple runs
for r=1:rounds

%GMM Initialization
%Initialize all three covariance matrices to the emperical covariance of
%the data
ecov=cov(x_train_data);
sig_m = cell(1,NGMM);
pi_k=zeros(1,NGMM);
for i=1:NGMM
    sig_m{i}=ecov;
    pi_k(i)=1/NGMM;
end

    
%Randomly initialize the means by sampling from a single multivariate
%Gaussian. Assume uniform mixing weights.
mudata = mean(x_train_data,1);
muinit = zeros(10,NGMM);

for i=1:NGMM
    muinit(:,i) = mvnrnd(mudata,ecov);
end

%% Part A

L_it=zeros(1,iteration);
for c=1:iteration

%For the sake of saving time, introduce inversed sig
sig_m_inv = cell(1,NGMM);
sig_m_det = cell(1,NGMM);
for m=1:NGMM
    sig_m_inv{m}=inv(sig_m{m});
    sig_m_det{m}=det(sig_m{m});
end


%Prepare variable for L
phi_den_xi=zeros(1,size(x_train_data,1));

%E-step
phi_i_k=zeros(NGMM,size(x_train_data,1));
for i=1:size(x_train_data,1)
    %Find the denominator for phi_i_k
    phi_den=0;
    for k=1:NGMM
        phi_den=phi_den+pi_k(k)*p_xmu_sig(transpose(x_train_data(i,:)),muinit(:,k),sig_m_det{k},sig_m_inv{k});
    end
    
    phi_den_xi(i)=phi_den;
    
    %K-loop to find the probabilty of in each Gaussian.
    for k=1:NGMM
        phi_i_k(k,i)=pi_k(k)*p_xmu_sig(transpose(x_train_data(i,:)),muinit(:,k),sig_m_det{k},sig_m_inv{k})/phi_den;
    end
end

%M-step
for k=1:NGMM
    %Count the number of cases
    n_k=sum(phi_i_k(k,:));
    %Update the mixing weight
    pi_k(k)=n_k/size(x_train_data,1);
    %Prep for updating mu and sig
    musumref = 0;
    sig_temp=zeros(10,10);
    for i=1:size(x_train_data,1)
        musumref = musumref + phi_i_k(k,i)*transpose(x_train_data(i,:));
    end
    %Update mu
    muinit(:,k)=1/n_k*musumref;
    %Update sigma
    for i=1:size(x_train_data,1)
        sig_temp = sig_temp + phi_i_k(k,i)*(transpose(x_train_data(i,:))-muinit(:,k))*transpose(transpose(x_train_data(i,:))-muinit(:,k));
    end
    sig_m{k}=1/n_k*sig_temp;
        
end

%Calculate value of L
L_it(c)=sum(log(phi_den_xi));

L_it_lst(r,:)=L_it;

if(c==30)
    if(d==1)
        pi_k_1_data{r}=pi_k;
        mu_1_data{r}=muinit;
        sig_1_data{r}=sig_m;
        L_it_1(r)=L_it(iteration);
        
    else
        pi_k_0_data{r}=pi_k;
        mu_0_data{r}=muinit;
        sig_0_data{r}=sig_m;
        L_it_0(r)=L_it(iteration);
    end
end

end

end

titlestring = {'EM Algorithm for GMM on Class 1','EM Algorithm for GMM on Class 0'};
x=linspace(1,iteration,iteration);
figure
hold on
title(titlestring{d});
xlabel('Iterations') 
ylabel('log L') 
for i=1:rounds
    plot(x,L_it_lst(i,:));
end
%legend({'Round 1','K=3','K=4','K=5'},'Location','northeast')
hold off

end

%Post computation data extraction
[MaxL_1,I1] = max(L_it_1);
[MaxL_0,I0] = max(L_it_0);

pi_k_1_max = pi_k_1_data{I1};
mu_1_max = mu_1_data{I1};
sig_1_max = sig_1_data{I1};

pi_k_0_max = pi_k_0_data{I0};
mu_0_max = mu_0_data{I0};
sig_0_max = sig_0_data{I0};
