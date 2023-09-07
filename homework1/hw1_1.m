clear; close all;
%Initialize Parameters and Generate Data
N = 10000; %Number of data points
n=4; %Dimensions of data
p0 = 0.35; %Prior for label 0
p1 = 0.65; %Prior for label 1
u = rand(1,N)>=p0; %Determine posteriors
%Create appropriate number of data points from each distribution
N0 = length(find(u==0));
N1 = length(find(u==1));
N=N0+N1;
label=[zeros(1,N0) ones(1,N1)];
%Parameters for two classes
mu0 = [-1/2;-1/2;-1/2;-1/2];
Sigma0 = [2,-0.5,0.3,0;
    -0.5,1,-0.5,0;
    0.3,-0.5,1,0;
    0,0,0,2];
mu1 = [1;1;1;1];
Sigma1 = [1,0.3,-0.2,0;
    0.3,2,0.3,0;
    -0.2,0.3,1,0;
    0,0,0,3];
%Generate data as prescribed in assignment description
r0 = mvnrnd(mu0, Sigma0, N0);
r1 = mvnrnd(mu1, Sigma1, N1);
%Plot data showing two classes
figure;
plot3(r0(:,1),r0(:,2),r0(:,3),'.b','DisplayName','Class 0');
axis equal;
hold on;
plot3(r1(:,1),r1(:,2),r1(:,3),'.r','DisplayName','Class 1');
axis equal;
hold on;
xlabel('x_1');ylabel('x_2');zlabel('x_3');
grid on;
title('Class 0 and Class 1 Data');
legend 'show';
%Combine data from each distribution into a single dataset
x=zeros(N,n);
x(label==0,:)=r0;
x(label==1,:)=r1;
%Part 1: ERM Classification with True Knowledge
discScore=log(evalGaussian(x' ,mu1,Sigma1)./evalGaussian(x' ,mu0,Sigma0));
sortDS=sort(discScore);
%Generate vector of gammas for parametric sweep
logGamma=[min(discScore)-eps sort(discScore)+eps];
for ind=1:length(logGamma)
    decision=discScore>logGamma(ind);
    Num_pos(ind)=sum(decision);
    pFP(ind)=sum(decision==1 & label==0)/N0;
    pTP(ind)=sum(decision==1 & label==1)/N1;
    pFN(ind)=sum(decision==0 & label==1)/N1;
    pTN(ind)=sum(decision==0 & label==0)/N0;
    %Two ways to make sure I did it right
    pFE(ind)=(sum(decision==0 & label==1) + sum(decision==1 & label==0))/N;
    pFE2(ind)=(pFP(ind)*N0 + pFN(ind)*N1)/N;
end
%Calculate Theoretical Minimum Error
logGamma_ideal=log(p0/p1);
decision_ideal=discScore>logGamma_ideal;
pFP_ideal=sum(decision_ideal==1 & label==0)/N0;
pTP_ideal=sum(decision_ideal==1 & label==1)/N1;
pFE_ideal=(pFP_ideal*N0+(1-pTP_ideal)*N1)/(N0+N1);
%Estimate Minimum Error
%If multiple minimums are found choose the one closest to the theoretical
%minimum
[min_pFE, min_pFE_ind]=min(pFE);
if length(min_pFE_ind)>1
    [~,minDistTheory_ind]=min(abs(logGamma(min_pFE_ind)-logGamma_ideal));
    min_pFE_ind=min_pFE_ind(minDistTheory_ind);
end
%Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(min_pFE_ind));
min_FP=pFP(min_pFE_ind);
min_TP=pTP(min_pFE_ind);
%Plot
figure;
plot(pFP,pTP,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP,min_TP,'o','DisplayName','Estimated Min. Error','LineWidth',2);
plot(pFP_ideal,pTP_ideal,'+','DisplayName',...
'Theoretical Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;
fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
exp(logGamma_ideal),100*pFE_ideal);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',minGAMMA,100*min_pFE);
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
%Part 2: Naive Bayesian Classifier
Sigma_NB=eye(4); %Assumed covariance
%Generate data to illustrate assumptions
r0_NB = mvnrnd(mu0, Sigma_NB, N0);
r1_NB = mvnrnd(mu1, Sigma_NB, N1);
%Plot Data demonstrating Naive Assumption
figure;
plot3(r0_NB(:,1),r0_NB(:,2),r0_NB(:,3),'.b','DisplayName','Class 0');
axis equal;
hold on;
plot3(r1_NB(:,1),r1_NB(:,2),r1_NB(:,3),'.r','DisplayName','Class 1');
axis equal;
xlabel('x_1');ylabel('x_2');zlabel('x_3');
grid on;
title('Assumed Class 0 and Class 1 Data Distributions for Naive Bayesian Classification');
legend 'show';
%Plot comparison of actual data to naive assumption
figure;
plot3(r0(:,1),r0(:,2),r0(:,3),'.b','DisplayName','Class 0 Actual Distribution');
hold on;
plot3(r0_NB(:,1),r0_NB(:,2),r0_NB(:,3),...
'.c','DisplayName','Class 0 Naive Assumption');
plot3(r1(:,1),r1(:,2),r1(:,3),'.r',...
'DisplayName','Class 1 Actual Distribution');
plot3(r1_NB(:,1),r1_NB(:,2),r1_NB(:,3),...
'.m','DisplayName','Class 1 Naive Assumption');
axis equal;
xlabel('x_1');ylabel('x_2');zlabel('x_3');
grid on;
title('Comparison of Actual Distribution to Naive Assumption');
legend 'show';
%Evaluate for different gammas
discScore_NB=...
log(evalGaussian(x' ,mu1,Sigma_NB)./evalGaussian(x' ,mu0,Sigma_NB));
logGamma_NB=[min(discScore_NB)-0.1 sort(discScore_NB)+0.1];
for ind=1:length(logGamma_NB)
    decision=discScore_NB>logGamma_NB(ind);
    Num_pos_NB(ind)=sum(decision);
    pFP_NB(ind)=sum(decision==1 & label==0)/N0;
    pTP_NB(ind)=sum(decision==1 & label==1)/N1;
    pFN_NB(ind)=sum(decision==0 & label==1)/N1;
    pTN_NB(ind)=sum(decision==0 & label==0)/N0;
    pFE_NB(ind)=(sum(decision==0 & label==1)...
    + sum(decision==1 & label==0))/(N0+N1);
    pFE2_NB(ind)=pFP(ind)*p0+pFN(ind)*p1;
end
%Estimated Minimum Error
[min_pFE_NB, min_pFE_ind_NB]=min(pFE_NB);
minGAMMA_NB=exp(logGamma(min_pFE_ind_NB));
min_FP_NB=pFP_NB(min_pFE_ind_NB);
min_TP_NB=pTP_NB(min_pFE_ind_NB);
%Plot Results
figure;
plot(pFP_NB,pTP_NB,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP_NB,min_TP_NB,'o','DisplayName',...
'Estimated Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve for Naive Bayesian');
legend 'show';
grid on; box on;
figure;
plot(logGamma_NB,pFE_NB,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma_NB(min_pFE_ind_NB),pFE_NB(min_pFE_ind_NB),'ro',...
'DisplayName','Minimum Error','LineWidth',2);
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma for Naive Bayesian Estimate')
grid on;
legend 'show';
fprintf('Estimated for NB: Gamma=%1.2f, Error=%1.2f%%\n',...
minGAMMA_NB,100*min_pFE_NB);
%Part 3: Fisher LDA
%Compute Sample Mean and covariances
mu0_hat=mean(r0)';
mu1_hat=mean(r1)';
Sigma0_hat=cov(r0);
Sigma1_hat=cov(r1);
%Compute scatter matrices
Sb=(mu0_hat-mu1_hat)*(mu0_hat-mu1_hat)';
Sw=Sigma0_hat+Sigma1_hat;
%Eigen decompostion to generate WLDA
[V,D]=eig(inv(Sw)*Sb);
[~,ind]=max(diag(D));
w=V(:,ind);
y=w'*x';
w=sign(mean(y(label==1)-mean(y(label==0))))*w;
y=sign(mean(y(label==1)-mean(y(label==0))))*y;
%Evaluate for different taus
tau=[min(y)-0.1 sort(y)+0.1];
for ind=1:length(tau)
    decision=y>tau(ind);
    Num_pos_LDA(ind)=sum(decision);
    pFP_LDA(ind)=sum(decision==1 & label==0)/N0;
    pTP_LDA(ind)=sum(decision==1 & label==1)/N1;
    pFN_LDA(ind)=sum(decision==0 & label==1)/N1;
    pTN_LDA(ind)=sum(decision==0 & label==0)/N0;
    pFE_LDA(ind)=(sum(decision==0 & label==1)...
    + sum(decision==1 & label==0))/(N0+N1);
end
%Estimated Minimum Error
[min_pFE_LDA, min_pFE_ind_LDA]=min(pFE_LDA);
minTAU_LDA=tau(min_pFE_ind_LDA);
min_FP_LDA=pFP_LDA(min_pFE_ind_LDA);
min_TP_LDA=pTP_LDA(min_pFE_ind_LDA);
%Plot results
figure;
plot(y(label==0),zeros(1,N0),'o','DisplayName','Label 0');
hold all;
plot(y(label==1),ones(1,N1),'o','DisplayName','Label 1');
ylim([-1 2]);
plot(repmat(tau(min_pFE_ind_LDA),1,2),ylim,'m--',...
'DisplayName','Tau for Min. Error','LineWidth',2);
grid on;
xlabel('y');
title('Fisher LDA Projection of Data');
legend 'show';
figure;
plot(pFP_LDA,pTP_LDA,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP_LDA,min_TP_LDA,'o','DisplayName',...
'Estimated Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;
figure;
plot(tau,pFE_LDA,'DisplayName','Errors','LineWidth',2);
hold on;
plot(tau(min_pFE_ind_LDA),pFE_LDA(min_pFE_ind_LDA),'ro',...
'DisplayName','Minimum Error','LineWidth',2);
xlabel('Tau');
ylabel('Proportion of Errors');
title('Probability of Error vs. Tau for Fisher LDA')
grid on;
legend 'show';
fprintf('Estimated for LDA: Tau=%1.2f, Error=%1.2f%%\n',...
minTAU_LDA,100*min_pFE_LDA);
%% ======================= Question 1: Functions ====================== %%
%reference g/Code/evalGaussian.m
function g = evalGaussian(x ,mu,Sigma)
%Evaluates the Gaussian pdf N(mu, Sigma ) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2); %coefficient
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);%exponent
g = C*exp(E); %finalgaussianevaluation
end