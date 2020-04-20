% Assignment3MatrixFactorizationPrecition.m
% Johnny C. Li jcl2222@columbia.edu
% Take the 10 run data and find the 10 closest movies to target. The 
% Assignment3MatrixFactorization.m must be ran first or Workspace from
% a session has to be imported to reduce run/debug time. 
%

% Notes
% Star Wars = 50
% My Fair Lady = 485
% Goodfellas = 182

%Targets
targets = [50,485,182];

%Import movie titles
movie_title = textscan(fopen('Prob3_movies.txt'),'%s','delimiter','\n');
movie_title = [movie_title{1,1}];

%Allocate space for distance calculation
distances = zeros(size(targets,2),size(movie_title,1));

for t=1:size(targets,2)
    for i=1:size(movie_title,1)
        %Find Euclidean distance
        distances(t,i) = norm(v_arr{targets(t)}-v_arr{i});
    end
end

%Allocate space for movies
ten_closest = cell(size(targets,2),11);

index = 1:size(movie_title,1);

%Sort the distance and compile a list of top 11 with first entry being the
%target.
for t=1:size(targets,2)
    temp_list = [distances(t,:).' index.'];
    sorted = sortrows(temp_list);
    
    for i=1:11
        ten_closest(t,i)=movie_title(sorted(i,2));
    end
    
end

