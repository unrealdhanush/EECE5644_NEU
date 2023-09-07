dim=2; % Dimension of data

%Define datasets used for training and validation
D.d20.N=20;
D.d200.N=200;
D.d2k.N=2000;
D.d10k.N=10e3;
dType=fieldnames(D);

p=[0.6 0.4]; %prior probabilities

%Label 0

mu0=[-1 -1 ;1 1]';
Sigma0(:,:,1)=[1 0;0 1];
Sigma0(:,:,2)=[1 0;0 1];
alpha0=[0.5 0.5];

%Label 1 
mu1=[-1 1 ;1 -1]';
Sigma1(:,:,1)=[1 0;0 1];
Sigma1(:,:,2)=[1 0;0 1];
alpha1=[0.5 0.5];
figure;

% Generate data

for index=1:length(dType)
    D.(dType{index}).x=zeros(dim,D.(dType{index}).N); %Initialize Data
    
    %Determine posteriors
    D.(dType{index}).labels = rand(1,D.(dType{index}).N)>=p(1);
    D.(dType{index}).N0=sum(~D.(dType{index}).labels);
    D.(dType{index}).N1=sum(D.(dType{index}).labels);
    
    D.(dType{index}).phat(1)=D.(dType{index}).N0/D.(dType{index}).N;
    D.(dType{index}).phat(2)=D.(dType{index}).N1/D.(dType{index}).N;
    
    [D.(dType{index}).x(:,~D.(dType{index}).labels),D.(dType{index}).dist(:,~D.(dType{index}).labels)]=randGMM(D.(dType{index}).N0,alpha0,mu0,Sigma0);
    [D.(dType{index}).x(:,D.(dType{index}).labels),D.(dType{index}).dist(:,D.(dType{index}).labels)]=randGMM(D.(dType{index}).N1,alpha1,mu1,Sigma1);
    
    subplot(2,2,index);
    plot(D.(dType{index}).x(1,~D.(dType{index}).labels),D.(dType{index}).x(2,~D.(dType{index}).labels),'b.','DisplayName','Class 0');
    hold all;
    plot(D.(dType{index}).x(1,D.(dType{index}).labels),D.(dType{index}).x(2,D.(dType{index}).labels),'r.','DisplayName','Class 1');
    
    grid on;
    xlabel('x1');ylabel('x2');
    title([num2str(D.(dType{index}).N) 'Samples From Both Classes']);
end

legend 'show';

%Part 1: Optimal Classifier

px0=evalGMM(D.d10k.x,alpha0,mu0,Sigma0);
px1=evalGMM(D.d10k.x ,alpha1,mu1,Sigma1);
discScore=log(px1./px0);
sortScore=sort(discScore);

% Generate vector of gammas

logGamma=[min(discScore)-eps sort(discScore)+eps];
prob=CalcProb(discScore,logGamma,D.d10k.labels,D.d10k.N0,D.d10k.N1,D.d10k.phat);
logGamma_ideal=log(p(1)/p(2));
decision_ideal=discScore>logGamma_ideal;
p10_ideal=sum(decision_ideal==1 & D.d10k.labels==0)/D.d10k.N0;
p11_ideal=sum(decision_ideal==1 & D.d10k.labels==1)/D.d10k.N1;
pFE_ideal=(p10_ideal*D.d10k.N0+(1-p11_ideal)*D.d10k.N1)/(D.d10k.N0+D.d10k.N1);

% Estimate Minimum Error

[prob.min_pFE, prob.min_pFE_ind]=min(prob.pFE);
if length(prob.min_pFE_ind)>1
    [~,minDistTheory_ind]=min(abs(logGamma(prob.min_pFE_ind)-logGamma_ideal));
    prob.min_pFE_ind=prob.min_pFE_ind(minDistTheory_ind);
end

% Find minimum gamma and corresponding false and true positive rates

minGAMMA=exp(logGamma(prob.min_pFE_ind));
prob.min_FP=prob.p10(prob.min_pFE_ind);
prob.min_TP=prob.p11(prob.min_pFE_ind);

plotROC(prob.p10,prob.p11,prob.min_FP,prob.min_TP);
hold all;
plot(p10_ideal,p11_ideal,'+','DisplayName','Ideal Minimum Error');
plotMinPFE(logGamma,prob.pFE,prob.min_pFE_ind);
plotDecisions(D.d10k.x,D.d10k.labels,decision_ideal);
plotERMContours(D.d10k.x,alpha0,mu0,Sigma0,alpha1,mu1,Sigma1,logGamma_ideal);

%Part 2: Maximum Likelihood Estimate

option=optimset('MaxFunEvals',3000,'MaxIter',1000);
for index=1:length(dType)
    lin.x=[ones(1,D.(dType{index}).N); D.(dType{index}).x];
    lin.init=zeros(dim+1,1);
    [lin.theta,lin.cost]=fminsearch(@(theta)(costFun(theta,lin.x,D.(dType{index}).labels)),lin.init,option);
    lin.discScore=lin.theta'*[ones(1,D.d10k.N); D.d10k.x];
    gamma=0;
    lin.prob=CalcProb(lin.discScore,gamma,D.d10k.labels,D.d10k.N0,D.d10k.N1,D.d10k.phat);
 
    plotDecisions(D.d10k.x,D.d10k.labels,lin.prob.decisions);
    lin.prob.pFE
    title(sprintf('Data and Classifier Decisions for Linear Logistic Fit\nProbability of Error=%1.1f%%',100*lin.prob.pFE));

    %Using a Quadratic Fit 
    quad.x=[ones(1,D.(dType{index}).N); D.(dType{index}).x;D.(dType{index}).x(1,:).^2;
        D.(dType{index}).x(1,:).*D.(dType{index}).x(2,:);
        D.(dType{index}).x(2,:).^2];
    quad.init= zeros(2*(dim+1),1);
    [quad.theta,quad.cost]=fminsearch(@(theta)(costFun(theta,quad.x,D.(dType{index}).labels)),quad.init,option);
    quad.xScore=[ones(1,D.d10k.N); D.d10k.x; D.d10k.x(1,:).^2;
        D.d10k.x(1,:).*D.d10k.x(2,:); D.d10k.x(2,:).^2];
    quad.discScore=quad.theta'*quad.xScore;
    gamma=0;
    quad.prob=CalcProb(quad.discScore,gamma,D.d10k.labels,D.d10k.N0,D.d10k.N1,D.d10k.phat);
    
    plotDecisions(D.d10k.x,D.d10k.labels,quad.prob.decisions);
    title(sprintf('Data and Classifier Decisions for Quad Logistic Fit\nProbability of Error=%1.1f%%',100*quad.prob.pFE));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cost=costFun(theta,x,labels)
h=1./(1+exp(-x'*theta));
cost=-1/length(h)*sum((labels'.*log(h)+(1-labels)'.*(log(1-h))));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
labels(ind)=m-1;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z = randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = evalGaussian(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob=CalcProb(discScore,logGamma,labels,N0,N1,phat)
for ind=1:length(logGamma)
prob.decisions=discScore>=logGamma(ind);
Num_pos(ind)=sum(prob.decisions);
prob.p10(ind)=sum(prob.decisions==1 & labels==0)/N0;
prob.p11(ind)=sum(prob.decisions==1 & labels==1)/N1;
prob.p01(ind)=sum(prob.decisions==0 & labels==1)/N1;
prob.p00(ind)=sum(prob.decisions==0 & labels==0)/N0;
prob.pFE(ind)=prob.p10(ind)*phat(1) + prob.p01(ind)*phat(2);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotROC(p10,p11,min_FP,min_TP)
figure;
plot(p10,p11,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP,min_TP,'o','DisplayName','Estimated Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotMinPFE(logGamma,pFE,min_pFE_ind)
figure;
plot(logGamma,pFE,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma(min_pFE_ind),pFE(min_pFE_ind),...
'ro','DisplayName','Minimum Error','LineWidth',2);
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma')
grid on;
legend 'show';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotDecisions(x,labels,decisions)
ind00 = find(decisions==0 & labels==0);
ind10 = find(decisions==1 & labels==0);
ind01 = find(decisions==0 & labels==1);
ind11 = find(decisions==1 & labels==1);
figure; % class 0 circle, class 1 +, correct green, incorrect red

plot(x(1,ind00),x(2,ind00),'og','DisplayName','Class 0, Correct'); hold on,
plot(x(1,ind10),x(2,ind10),'or','DisplayName','Class 0, Incorrect'); hold on,
plot(x(1,ind01),x(2,ind01),'+r','DisplayName','Class 1, Correct'); hold on,
plot(x(1,ind11),x(2,ind11),'+g','DisplayName','Class 1, Incorrect'); hold on,
axis equal,
grid on;
title('Data and their classifier decisions versus true labels');
xlabel('x_1'), ylabel('x_2');
legend('Correct decisions for data from Class 0',...
'Wrong decisions for data from Class 0',...
'Wrong decisions for data from Class 1',...
'Correct decisions for data from Class 1');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotERMContours(x,alpha0,mu0,Sigma0,alpha1,mu1,Sigma1,logGamma_ideal)
horizGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
vertGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizGrid,vertGrid);
discrimScoreGridVals =log(evalGMM([h(:)';v(:)'],alpha1,mu1,Sigma1))-log(evalGMM([h(:)';v(:)'],alpha0,mu0,Sigma0)) - logGamma_ideal;
minDSGV = min(discrimScoreGridVals);
maxDSGV = max(discrimScoreGridVals);
discrimScoreGrid = reshape(discrimScoreGridVals,91,101);
contour(horizGrid,vertGrid,discrimScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours
% including the contour at level 0
legend('Correct decisions - Class 0',...
    'Incorrect decisions - Class 0',...
    'Incorrect decisions - Class 1',...
    'Correct decisions - Class 1',...
    'Equilevel contours of discriminant function' ),
end