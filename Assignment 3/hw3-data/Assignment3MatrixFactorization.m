% Assignment3MatrixFactorization.m
% Johnny C. Li jcl2222@columbia.edu
% Implement the MAP inference algorithm for the matrix completion problem.
%

%% Constant parameter definition
sigsq = 0.25;
d = 10;
lam = 1;
iteration = 100;
runs =10;
I=eye(d);

%% Import Data and Format
raw_train_data = csvread('Prob3_ratings.csv',0,0);

N1=max(raw_train_data(:,1));
N2=max(raw_train_data(:,2));

%Initialize matrix (User ID (R) by Movie ID (C))
rating_matrix_train = zeros(N1,N2);

%Populate the rating matrix
for i=1:size(raw_train_data,1)
    rating_matrix_train(raw_train_data(i,1),raw_train_data(i,2))=raw_train_data(i,3);
end

%Import test data
raw_test_data = csvread('Prob3_ratings_test.csv',0,0);
ds = size(raw_test_data,1);
%Prediction result
MF_pred = zeros(ds,1);
%Final L and RMSE table
L_RMSE_tbl = zeros(runs,3);

%% Being Matrix Factorization

%Define u and v arrays
u_arr = cell(N1,1);
v_arr = cell(1,N2);

%Log Likelihood objective
L_its=zeros(runs,iteration);

%wb = waitbar(0,'Initialization');

for r=1:runs
    %Initialize u_i as row
    for i=1:N1
        u_arr{i}=mvnrnd(zeros(1,d),I);
    end
    
    %Initialize v_j as column
    for j=1:N2
        v_arr{j}=mvnrnd(zeros(d,1),I).';
    end
    
    %Misc
    RMSE_sum = 0;
    
    
    %% Start iteration
    for it=1:iteration
        %waitbar((it+iteration*(r-1))/(iteration*runs),wb,['Run ' num2str(r) ' Iteration ' num2str(it)]);
        %Storage for L
        L_1 = 0;
        L_2 = 0;
        L_3 = 0;
        %Update user location
        for i=1:N1
            v_j_dot_vjt_sum = zeros(d,d);
            v_j_times_rating_sum = zeros(d,1);
            %Search for user rated objects and perform v_j*v_j.' and the
            %sum for rating*vector.
            for j=1:N2
                if(rating_matrix_train(i,j)~=0)
                    v_j_dot_vjt_sum = v_j_dot_vjt_sum +  v_arr{j}*v_arr{j}.';
                    v_j_times_rating_sum = v_j_times_rating_sum + rating_matrix_train(i,j)*v_arr{j};
                end
            end
            u_arr{i}=(lam*sigsq*I+v_j_dot_vjt_sum)\v_j_times_rating_sum;
            L_2 = L_2 + lam/2*norm(u_arr{i})^2;
        end
        %Update object(movie) location
        for j=1:N2
            u_i_dot_uit_sum = zeros(d,d);
            u_i_times_rating_sum = zeros(d,1);
            %Search for user rated objects and perform v_j*v_j.' and the
            %sum for rating*vector.
            for i=1:N1
                if(rating_matrix_train(i,j)~=0)
                    u_i_dot_uit_sum = u_i_dot_uit_sum +  u_arr{i}*u_arr{i}.';
                    u_i_times_rating_sum = u_i_times_rating_sum + rating_matrix_train(i,j)*u_arr{i};
                end
            end
            v_arr{j}=(lam*sigsq*I+u_i_dot_uit_sum)\u_i_times_rating_sum;
            L_3 = L_3 + lam/2*norm(v_arr{j})^2;
        end

        %Calculatet the L per iteration             
        for i=1:N1
            L_2 = L_2 + lam/2*norm(u_arr{i})^2;
            for j=1:N2
               if(rating_matrix_train(i,j)~=0)
                   L_1 = L_1 + 1/(2*sigsq)*norm(rating_matrix_train(i,j)-u_arr{i}.'*v_arr{j})^2;
               end
            end
        end
            
        L_its(r,it)=-L_1-L_2-L_3;
    end
    
    %Calculate predicted rating and RMSE
    for i=1:ds
        MF_pred(i)=u_arr{raw_test_data(i,1)}.'*v_arr{raw_test_data(i,2)};
        RMSE_sum = RMSE_sum + (MF_pred(i)-raw_test_data(i,3))^2/ds;
    end
    
    %Create table
    L_RMSE_tbl(r,1) = L_its(r,iteration);
    L_RMSE_tbl(r,2) = sqrt(RMSE_sum);
    L_RMSE_tbl(r,3) = r;
    
end

    sorted_L_RMSE = sortrows(L_RMSE_tbl);
    

%% Plot
x=linspace(1,iteration,iteration);
figure
hold on
title('Log Joint Likelihood vs Iteration');
xlabel('Iterations') 
ylabel('L') 
for i=1:runs
    plot(x,L_its(i,:));
end
hold off
