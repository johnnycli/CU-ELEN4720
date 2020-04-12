% Assignment3MatrixFactorizationPrecition.m
% Johnny C. Li jcl2222@columbia.edu
% Take the 10 run data and perform prediction. Calculate the RMSE.
% Import data to workspace is required.
%

%Import test data
raw_test_data = csvread('Prob3_ratings_test.csv',0,0);

ds = size(raw_test_data,1);

%Prediction result
MF_pred = zeros(ds,1);

RMSE_sum = 0;

%Calculate predicted rating and RMSE
for i=1:ds
    MF_pred(i)=u_arr{raw_test_data(i,1)}.'*v_arr{raw_test_data(i,2)};
    RMSE_sum = RMSE_sum + (MF_pred(i)-raw_test_data(i,3))^2/ds;
end

RMSE = sqrt(RMSE_sum);