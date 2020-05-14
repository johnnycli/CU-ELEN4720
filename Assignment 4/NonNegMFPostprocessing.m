% NonNegMFPostprocessing.m
% Johnny C. Li jcl2222@columbia.edu
% 
% Data postprocessing for NonNegMF.m. Workspace or a run through of .m 
% is required.
%

%% Import Data
fid = fopen('nyt_vocab.dat');
vocab_list = textscan(fid,'%s','delimiter','\n');
vocab_list= vocab_list{1,1};
fclose(fid);

%% Plot D(X||WH) against iterations for problem 2a.
figure
hold on
title('Non-negative Matrix Factorization on 25 Topics');
xlabel('Iterations') 
ylabel('D(X||WH)') 
plot(L(1,:),L(2,:));
hold off


%% P2b
% Normalize columns of W.
W_norm = zeros(3012,25);
word_index = 1:3012;
word_list = cell(10,25);
weight_list = zeros(10,25);
for i = 1:25
    W_norm(:,i) = W(:,i)/sum(W(:,i));
    sorted = sortrows([W_norm(:,i) word_index.' ],'descend');
    
    for j=1:10
        word_list{j,i} = vocab_list{sorted(j,2)};
        weight_list(j,i) = sorted(j,1);
    end
end







