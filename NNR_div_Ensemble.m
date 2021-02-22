function value  = NNR_div_Ensemble(Z,N,I,alpha) 
% I is the type of Divergence: 1=KL,2=Renyi. For Renyi divergence, alpha is
% the parameter
% Z is 2*N by d vector, X=Z(1:N) and Y=Z(N+1:2*N)

d=length(Z(1,:));  %dim of dataset
L=2;
l1=1:L;

%% CVX version for calculating the optimal weight

% ODin1
psi1=zeros(L,d);
for r=1:d
    psi1(:,r)=l1.^(r);
end
psi1(:,d+1)=l1.^(-d);

cvx_begin quiet
    variables w1(L) t(1);
    minimize t
    subject to
        sum(w1)==1;
        norm(w1,2)<=t;
        max([N.^(.5-(1:d)/(2*d)) 1].*abs(w1'*psi1))<=t;

cvx_end


w1;




%% Estimate
temp_kern1=zeros(L,1);

for i=1:L
    k=fix(sqrt(N)*i);
    [IDX,D] = knnsearch(Z,Z,'k',k+1); %IDX is a Matrix, rows are different nodes and culomns are indeces of KNNs. (The forst index is the point itsef)
    Temp= (IDX<=N); % For each row (node) obtain how many of KNN is of the set X (those who have IDX=<N)
    Temp2=sum(Temp,2); % Temp2 is the number of indices from X set.

    % KL-Divergence
    if I==1 
       Rat=(k-Temp2+1)./(Temp2+1);
       Temp3=log(Rat(N+1:2*N,1));
       value=sum(Temp3)/N; % Average over KNN ratios of Y set
    end

    % Renyi Divergence
    if I==2 
       Rat=(Temp2)./(k-Temp2+1);
       Temp3=Rat(N+1:2*N,1).^alpha;
       value=sum(Temp3)/N; % Average over KNN ratios of Y set
       value= 1/(alpha-1)*log(value); % Renyi Divergence
    end
    temp_kern1(i,1)=value;
end

%% Ensemble estimator
temp_kern1;

div=max(w1'*temp_kern1,0);