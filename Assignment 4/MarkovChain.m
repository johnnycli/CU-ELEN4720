% MarkovChain.m
% Johnny C. Li jcl2222@columbia.edu
% 
% Rank 769 college football teams based on the scores of every game in the 2019
% season.
%

%% Import Data

% Data follows the format:
% Team A index, Team A points, Team B index, Team B points
score_data = csvread('CFB2019_scores.csv',0,0);

team_name = textscan(fopen('TeamNames.txt'),'%s','delimiter','\n');
team_name= team_name{1,1};

M_hat = zeros(769,769);

%% Look thorugh the score data and populate the W_hat matrix.
for i = 1:size(score_data,1)
    % If Team A has more points than Team B.
    if(score_data(i,2)>score_data(i,4))
        % Update M_ii
        M_hat(score_data(i,1),score_data(i,1)) = M_hat(score_data(i,1),score_data(i,1)) + 1 + score_data(i,2)/(score_data(i,2)+score_data(i,4)); 
        % Update M_ji transistion. (State j to i)
        M_hat(score_data(i,3),score_data(i,1)) = M_hat(score_data(i,3),score_data(i,1)) + 1 + score_data(i,2)/(score_data(i,2)+score_data(i,4)); 
        
    % If Team B has more points than Team A.
    elseif(score_data(i,2)<score_data(i,4))
        % Update M_jj
        M_hat(score_data(i,3),score_data(i,3)) = M_hat(score_data(i,3),score_data(i,3)) + 1 + score_data(i,4)/(score_data(i,2)+score_data(i,4)); 
        % Update M_ij transistion. (State i to j)
        M_hat(score_data(i,1),score_data(i,3)) = M_hat(score_data(i,1),score_data(i,3)) + 1 + score_data(i,4)/(score_data(i,2)+score_data(i,4));      
    end
end

%% Create M by normalizing M_hat and define w_0, w_t
M = zeros(769,769);
for i = 1:size(M_hat,1)
    M(i,:) = M_hat(i,:)/sum(M_hat(i,:));
end

%% Find w_inf
% Find the eigenvectors and eigenvalues of M transposed.
[revec,eval,evec] = eig(M.'); 
w_inf = evec(:,1).';
w_inf = w_inf/sum(w_inf);

% Initialize w_t as uniformly distributed w_0
w_t = rand(1,769);
team_index = 1:size(team_name,1);
top_teams = cell(25,4);
% Used for storing w_t - w_inf
wdiff = zeros(10000,1);
%% Step through the chain and print top 25 teams at t=10, 100, 1000, 10000.
for t=1:10000
    % Perform w_t update
    w_t = w_t*M;
    
    % Calculate difference
    wdiff(t) = sum(w_t-w_inf);
    
    % Print top teams at given t
    if(t==10 || t==100 || t==1000 || t==10000)
        w_temp = [w_t.' team_index.'];
        % Sort in descend order
        sorted = sortrows(w_temp,'descend');
        for i=1:25
            % Store team order at t.
            top_teams{i,log10(t)}=team_name{sorted(i,2)};
        end
    end
end

%% Plot
x=linspace(1,10000,10000);
figure
hold on
title('|w_{t} - w_{\infty}|_{1}');
xlabel('t') 
ylabel('|w_{t} - w_{\infty}|_{1}') 
plot(x,wdiff);
hold off


