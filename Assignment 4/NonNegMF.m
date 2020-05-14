% NonNegMF.m
% Johnny C. Li jcl2222@columbia.edu
% 
% Factorize an N × M matrix X into a rank-K approximation W H, where W is
% N × K, H is K × M and all values in the matrices are nonnegative.
%

%% Import Data
fid = fopen('nyt_vocab.dat');
vocab_list = textscan(fid,'%s','delimiter','\n');
vocab_list= vocab_list{1,1};
fclose(fid);

% Import the text file. Formatted by Data_1, Data_2, ... \n
% Data_n follows the format word_index:word_count
fid = fopen('nyt_data.txt');
raw_data = textscan( fid, repmat('%f',[1,1766]), 'Delimiter',{':',','}, 'EmptyValue',NaN );
raw_data = cell2mat(raw_data);
fclose(fid);

% Pre-define space for X.
X = zeros(3012,8447);

% Populate X based on raw data.
for j = 1:8447
    % Skip by 2 as the middle value is the word count.
    for i = 1:2:1766
      % If we hit NaN, the row is over. Move to the next.
      if(isnan(raw_data(j,i)))
        break;
      else
        X(raw_data(j,i),j) = raw_data(j,i+1);   
      end    
    end
end

%% Implement and run the NMF algorithm using the divergence penalty

% Model setups
K = 25;
iteration = 100;

% Randomly initialize W and H with non-negative values.
W = randi([0,6],3012,K);
H = randi([0,6],K,8447);

L = zeros(2,iteration);

%% Begin iterations
for it = 1:iteration
    fprintf('---------Iteration %d---------\n',it);
    %% Update values in H
    for k = 1:K
            denom = sum(W(:,k))+10e-16;
        for j = 1:8447
            num = W(:,k).*X(:,j)./(W*H(:,j)+10e-16);
            snum = sum(num);
            H(k,j) = H(k,j)*snum/denom;
        end
        
        % For debugging purposes.
        fprintf('Updating H. k = %d\n',k);
    end

    %% Update values in W
    for k=1:K
            denom = sum(H(k,:))+10e-16;
        for i= 1:3012
            num = H(k,:).*X(i,:)./(W(i,:)*H+10e-16);
            snum = sum(num);
            W(i,k) = W(i,k)*snum/denom;
        end

        % For debugging purposes.
        fprintf('Updating W. k = %d\n',k);

    end
    
    %% Find objective
    WH = W*H;
    L_temp = -(X.*log(WH)-WH);
    L_temp = nansum(L_temp,'all');
    L(1,it)=it;
    L(2,it)=L_temp;
end












