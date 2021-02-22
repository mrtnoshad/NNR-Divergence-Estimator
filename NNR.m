function value  = NNR(Z,N,k,I,alpha) 
% Z =[X;Y] where X and Y both are matrices with N rows and d columns. 
% N is the number of samples in X and Y. and d is dimension   
% An Example is:
%%%%%%%%%%%%
%R1 = mvnrnd(MU1,SIGMA1,N);
%R2 = mvnrnd(MU2,SIGMA2,N);
%Z= [R1;R2 ];
%%%%%%%%%%%%
% I determines the type of Divergence: I=1: KL-Divergence ,
% I=2: Renyi-Divergence
% alpha is the parameter for Renyi divergence, if it is KL-Divergence,
% alpha is not important



[IDX,D] = knnsearch(Z,Z,'k',k+1); %IDX is a Matrix, rows are different nodes and culomns are indeces of KNNs. (The forst index is the point itsef, so we take k+1 nearest neighbors)

Temp= (IDX<=N); % For each row (node) obtain how many of KNN is of the set X (those who have IDX<N)

Temp2=sum(Temp,2); % Temp2 is the number of indices from X set.
%Temp3=(Temp2./(k-Temp2+1)).^(alpha); %Ratios 
[k-Temp2 (Temp2+1)];

% KL-Divergence
if I==1 
    Rat=(k-Temp2+1)./(Temp2+1);
    Temp3=log(Rat(N+1:2*N,1));
end

% Renyi Divergence
if I==2 
    Rat=(Temp2)./(k-Temp2+1);
    %size(Rat)
    Temp3=Rat(N+1:2*N,1).^alpha;;
end

% f-Divergence
if I==3 
    Rat=(Temp2)./(k-Temp2+1);
    Temp3=(Rat(N+1:2*N,1).^2)-1;
end


value=sum(Temp3)/N; % Average over KNN ratios of Y set
if I==2 
    value= 1/(alpha-1)*log(value); % Renyi Divergence
end

% Non-zero:
value=max(0,value);

