clear;
close all;
N = 10000;
%N=1000; % taking the values and trying each time with the N as
%10,100,1000and 1000
%N=100;
%N=10;
delta = 1e0; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
%Replicating it 30 times
% Generate samples from a 4-component GMM
alpha_true = [0.2,0.3,0.4,0.1];
mu_true = [-10 10 10 -10;10 10 -10 -10];
Sigma_true(:,:,1) = [20 1;10 3];
Sigma_true(:,:,2) = [7 1;1 2];
Sigma_true(:,:,3) = [4 10;1 16];
Sigma_true(:,:,4) = [2 1;1 7];
x = randGMM(N,alpha_true,mu_true,Sigma_true);
figure(1);
figure(1),scatter(x(1,:),x(2,:),'ob'), hold on,
figure(1),legend('sample')
d = 2;
K = 10;
dummy = ceil(linspace(0,N,K+1));
for k = 1:K
indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end
avgp = zeros(1,6);
for M = 1:6
psum = zeros(1,10);
for k = 1:K
    indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
    xValidate = x(:,indValidate); % Using folk k as validation set
    if k == 1
        indTrain = [indPartitionLimits(k,2)+1:N];
    elseif k == K
        indTrain = [1:indPartitionLimits(k,1)-1];
else
        indTrain = [[1:indPartitionLimits(k-1,2)],[indPartitionLimits(k+1,1):N]];
end
    xTrain = x(:,indTrain); % using all other folds as training set
    Ntrain = length(indTrain);
    Nvalidate = length(indValidate);
    [alpha,mu,Sigma] = EMforGMM(Ntrain,xTrain,M,d,delta,regWeight);% determine dimensionality of samples and number of GMM components
    p = zeros(1,Nvalidate);
    for j = 1:Nvalidate
        for i = 1:M
            p(j) = p(j) +alpha(i)*evalGaussian(xValidate(:,j),mu(:,i),Sigma(:,:,i));
end
        p(j) = log(p(j));
end
    psum(k) = sum(p);
    dummy(k,M)=sum(p);
end
avgp(M) = sum(psum)/10;
if (avgp(M)== -inf)
    avgp(M) = -1e5;
end 
end
%Below code is by trying the fitgmdsit  and Replicating by 30 times
%for m=1:6
%for i=1:K
%[train, test]= kfld(X,i);%gives training and test data for a fold
% model_gmm = fitgmdist(train,m);%fitting a gmm model on train data for m components
% prior=model_gmm.ComponentProportion;%priors
% mean1=model_gmm.mu;%mean
% mean=mean1';
% cov=model_gmm.Sigma;%covariance
% logLikelihood(i) = sum(log(evalGMM(test,prior,mean,cov)));
%end
%final_mean(m)=sum(logLikelihood)/10;%Final probability for all 6 Components
%end
figure(2),scatter([1,2,3,4,5,6],avgp),set(gca,'yscale','log'),
figure(2),legend('order'),title('Orders log-liklihood '),
xlabel('order'), ylabel('logp')
function [alpha,mu,Sigma] = EMforGMM(N,x,M,d,delta,regWeight)
% Initialize the GMM to randomly selected samples
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the
nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 0; % Not converged at the beginning
while ~Converged
for l = 1:M
    temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
end
plgivenx = temp./sum(temp,1);
alphaNew = mean(plgivenx,2);
w = plgivenx./repmat(sum(plgivenx,2),1,N);
muNew = x*w';
for l = 1:M
    v = x-repmat(muNew(:,l),1,N);
    u = repmat(w(l,:),d,1).*v;
    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
end
Dalpha = sum(abs(alphaNew-alpha'));
Dmu = sum(sum(abs(muNew-mu)));
DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
t = t+1;
end 
end
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end 
end
%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end