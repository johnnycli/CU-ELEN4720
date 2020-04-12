% Assignment3Kmeans.m
% Johnny C. Li jcl2222@columbia.edu
% Implement the K-means algorithm to generate 500 observations from a 
% mixture of three Gaussians on R^2.
%


%% INITIAL DEFINITIONS
%--------------------------------------------------------------------------
pi=[0.2,0.5,0.3]; %This is actually not used.
mu = [0,3,0;0,0,3];
sig = eye(2);
K1=[2,3,4,5];
K2=[3,5];
%--------------------------------------------------------------------------

cdata = zeros(1,500);
xdata = zeros(2,500);

%% Generte 500 Observations from the mixed model
%--------------------------------------------------------------------------
for i=1:500
    %Generate random cluster.
    rtemp=randi(10);
    if(rtemp<=2)
        cdata(i)=1;
    elseif(rtemp>=8)
        cdata(i)=3;
    else
        cdata(i)=2;
    end
    
    %Generate xi based on Normal parameters
    genx=mvnrnd(mu(:,cdata(i)),eye(2));
    xdata(1,i)=genx(1);
    xdata(2,i)=genx(2);
end
%--------------------------------------------------------------------------

%% Part b Setup. Data preservation.
k3c1=[];
            k3c2=[];
            k3c3=[];
            k3mu=[];
            
            k5c1=[];
            k5c2=[];
            k5c3=[];
            k5c4=[];
            k5c5=[];
            k5mu=[];



%% Part a - 20 Iteration Objective 
%--------------------------------------------------------------------------
%Start the K loops
L_data=zeros(length(K1),20);
for kloop=1:length(K1)
        kval=K1(kloop);
       
        %Initialize K mu's randomly
        muinit = zeros(2,kval);
        for i=1:kval
            %This range should cover 3sig range.
            muinit(1,i)=rand*9-3;
            muinit(2,i)=rand*9-3;
        end
        
        %Visual Check
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        clf
        hold on
        for i=1:kval
                scatter(muinit(1,i),muinit(2,i),'filled','kd')
        end
        scatter(xdata(1,:),xdata(2,:));
        hold off
        pause(2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Start the 20 iteration loops
        for c=1:20
            %Step 1: Update c based on mu
            ci_updated = zeros(1,500);
            for j=1:500
                %At x_j calculate the distance from all clusters
                dist_calc_ci = zeros(1,kval);
                for i=1:kval
                    dist_calc_ci(i)=norm(xdata(:,j)-muinit(:,i));
                end
                %Assign the cluster number based on min distance.
                [Mval,Indx]=min(dist_calc_ci);
                ci_updated(j)=Indx;
            end
            %Step 1: Update mu_k
            for i=1:kval
                nk=0;
                rsum=zeros(2,1);
                for j=1:500 
                    %Counter for nk
                    if(ci_updated(j)==i)
                        nk=nk+1;
                        rsum=rsum+xdata(:,j);
                    end
                end
                muinit(1,i)=1/nk*rsum(1);
                muinit(2,i)=1/nk*rsum(2);
            end
            
            %Sum the L2 distance for all ck=updated_k
            %Calculate L for this iteration
            L_iteration=0;
            for j=1:500 
                for i=1:kval
                    if(ci_updated(j)==i)
                        L_iteration = L_iteration+norm(xdata(:,j)-muinit(:,i));
                    end
                end
            end
            L_data(kloop,c)=L_iteration;
            
            
            %This part is used to visualize (place breakpoint)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            temp1=[];
            temp2=[];
            temp3=[];
            temp4=[];
            temp5=[];
            for n=1:500
                if (ci_updated(n)==1)
                    temp1=[temp1 xdata(:,n)];
                elseif (ci_updated(n)==2)
                    temp2=[temp2 xdata(:,n)];
                elseif (ci_updated(n)==3)
                    temp3=[temp3 xdata(:,n)];
                elseif (ci_updated(n)==4)
                    temp4=[temp4 xdata(:,n)];
                else
                    temp5=[temp5 xdata(:,n)];
                    
                end
            end
            temp1=temp1.';
            temp2=temp2.';
            temp3=temp3.';
            temp4=temp4.';
            temp5=temp5.';
            clf
            hold on          
            for i=1:kval
                scatter(muinit(1,i),muinit(2,i),'filled','kd')
            end
            if(size(temp1)>=1)
            scatter(temp1(:,1),temp1(:,2),'r');
            end
            if(size(temp2)>=1)
            scatter(temp2(:,1),temp2(:,2),'g');
            end
            if(size(temp3)>=1)
                scatter(temp3(:,1),temp3(:,2),'b');
            end
            if(size(temp4)>=1)
                scatter(temp4(:,1),temp4(:,2),'y');
            end
            if(size(temp5)>=1)
                scatter(temp5(:,1),temp5(:,2),'c');
            end
            hold off
            
            %Preserve Data for K=3 and K=5 for plotting purposes.            
            if(kval==3&&c==20)
                k3c1=temp1;
                k3c2=temp2;
                k3c3=temp3;
                k3mu=muinit;
            end
            
            if(kval==5&&c==20)
                k5c1=temp1;
                k5c2=temp2;
                k5c3=temp3;
                k5c4=temp4;
                k5c5=temp5;
                k5mu=muinit;
            end
            
            
            pause(0.1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
        end
        
end

%Plotting results
x=linspace(1,20,20);
figure
hold on
title('K-Mean Objective Minimization');
xlabel('Iterations') 
ylabel('L') 
plot(x,L_data(1,:));
plot(x,L_data(2,:));
plot(x,L_data(3,:));
plot(x,L_data(4,:));
legend({'K=2','K=3','K=4','K=5'},'Location','northeast')
hold off

%--------------------------------------------------------------------------

%% Part b - 50 Iteration Clusters 
%--------------------------------------------------------------------------
figure
hold on
title('K-Mean Clustering for K=3');
xlabel('x_1') 
ylabel('x_2') 
for i=1:3
  scatter(k3mu(1,i),k3mu(2,i),'filled','kd')
  if(size(k3c1)>=1)
            scatter(k3c1(:,1),k3c1(:,2),'r');
  end
  if(size(k3c2)>=1)
            scatter(k3c2(:,1),k3c2(:,2),'g');
  end
  if(size(k3c3)>=1)
            scatter(k3c3(:,1),k3c3(:,2),'b');
  end
end
hold off

figure
hold on
title('K-Mean Clustering for K=5');
xlabel('x_1') 
ylabel('x_2') 
for i=1:5
  scatter(k5mu(1,i),k5mu(2,i),'filled','kd')
  if(size(k5c1)>=1)
            scatter(k5c1(:,1),k5c1(:,2),'r');
  end
  if(size(k5c2)>=1)
            scatter(k5c2(:,1),k5c2(:,2),'g');
  end
  if(size(k5c3)>=1)
            scatter(k5c3(:,1),k5c3(:,2),'b');
  end
  if(size(k5c4)>=1)
            scatter(k5c4(:,1),k5c4(:,2),'y');
  end
  if(size(k5c5)>=1)
            scatter(k5c5(:,1),k5c5(:,2),'c');
  end
end
hold off

%--------------------------------------------------------------------------