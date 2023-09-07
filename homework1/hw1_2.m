clear;close all;clc;
n=3; %dimensions
N=10000; %samples

% Class means and covariances
mu(:,1) = [0; 0; 0];
mu(:,2) = [8; 8; 8];
mu3(:,1) = [0; 8; 8];
mu3(:,2) = [8; 0; 8];

Sigma(:,:,1)=[5 1 1; 1 5 1; 1 1 5];
Sigma(:,:,2)=[5 1 1; 1 5 1; 1 1 5];
Sigma3(:,:,1)=[5 1 1; 1 5 1; 1 1 5];
Sigma3(:,:,2)=[5 1 1; 1 5 1; 1 1 5];


% Class priors and true label
prior = [0.3 0.3 0.4];
x=zeros(n,N);
label=zeros(1,N);
for i=1:N
    r=rand(1);
    if r <= 0.3
        label(i)=1;
    elseif (0.3<r)&&(r<=0.6)
        label(i)=2;
    else
        label(i)=3;
    end
end
Nc=[sum(label==1),sum(label==2),sum(label==3)];

% Generate data as prescribed in assignment description
x(:,label==1)=randGMM(Nc(1),1,mu(:,1),Sigma(:,:,1));
x(:,label==2)=randGMM(Nc(2),1,mu(:,2),Sigma(:,:,2));
x(:,label==3)=randGMM(Nc(3),[0.5 0.5],mu3,Sigma3);

% Plot true class label
figure(11);
X = x(1, label==1);
Y = x(2, label==1);
Z = x(3, label==1);
scatter3(X,Y,Z,'r','filled');
hold on;
X = x(1, label==2);
Y = x(2, label==2);
Z = x(3, label==2);
scatter3(X,Y,Z,'g','filled');
hold on;
X = x(1, label==3);
Y = x(2, label==3);
Z = x(3, label==3);
scatter3(X,Y,Z,'b','filled');
title('Selected Gaussian PDF Samples');
legend('Class 1','Class 2','Class 3');
xlabel('x1');
ylabel('x2');
zlabel('x3');
hold off;

%%========================Part A========================%%
% Probabilities and class posteriors
pxgivenl(1,:)=evalGaussian(x,mu(:,1), Sigma(:,:,1));
pxgivenl(2,:)=evalGaussian(x,mu(:,2), Sigma(:,:,2));
pxgivenl(3,:)=evalGMM(x,[0.5 0.5],mu3,Sigma3);
px=prior*pxgivenl;
plgivenx=pxgivenl.*repmat(prior',1,N)./repmat(px,3,1); % Bayes theorem

% 0-1 loss matrix, expected risks, decision
lossMatrix=ones(3,3)-eye(3);
[decision,confusionMatrix]=runClassif(lossMatrix, plgivenx, label, Nc);

% Expected risk
estRisk = expRiskEstimate(lossMatrix, decision, label, N, 3);

% Confusion matrix
conf_mat = [sum(decision(label==1)==1) sum(decision(label==2)==1) sum(decision(label==3)==1); ...
               sum(decision(label==1)==2) sum(decision(label==2)==2) sum(decision(label==3)==2);
               sum(decision(label==1)==3) sum(decision(label==2)==3) sum(decision(label==3)==3)] ./ [sum(label==1) sum(label==2) sum(label==3)];           
figure(12)
h = heatmap(conf_mat);
h.Title = 'Confusion Matrix';

% Plot samples with marked correct & incorrect decision
figure(13);
plot3(x(1,label==1&decision==1), ...
    x(2,label==1&decision==1), ...
    x(3,label==1&decision==1), strcat('go'));
axis equal;
hold on;
plot3(x(1,label==1&decision~=1), ...
    x(2,label==1&decision~=1), ...
    x(3,label==1&decision~=1), strcat('ro'));
axis equal;
hold on;
plot3(x(1,label==2&decision==2), ...
    x(2,label==2&decision==2), ...
    x(3,label==2&decision==2), strcat('gx'));
axis equal;
hold on;
plot3(x(1,label==2&decision~=2), ...
    x(2,label==2&decision~=2), ...
    x(3,label==2&decision~=2), strcat('rx'));
axis equal;
hold on;
plot3(x(1,label==3&decision==3), ...
    x(2,label==3&decision==3), ...
    x(3,label==3&decision==3), strcat('gd'));
axis equal;
hold on;
plot3(x(1,label==3&decision~=3), ...
    x(2,label==3&decision~=3), ...
    x(3,label==3&decision~=3), strcat('rd'));
axis equal;
hold on;
grid on
xlabel('x1');ylabel('x2');zlabel('x3');
legend('Class 1 Correct', 'Class 1 Incorrect', 'Class 2 Correct', ...
    'Class 2 Incorrect','Class 3 Correct','Class 3 Incorrect');
hold off;
title('0-1 loss classification correctness');

%%========================Part B========================%%
% Loss matrix A10
lossMatrix10 = [0 1 10; 1 0 10; 1 1 0];
[decision10,confusionMatrix10]=runClassif(lossMatrix10, plgivenx, label, Nc);

% Expected risk 10
estRisk10=expRiskEstimate(lossMatrix10, decision10, label, N, 3);

% Confusion matrix for A10
conf_mat_10 = [sum(decision10(label==1)==1) sum(decision10(label==2)==1) sum(decision10(label==3)==1); ...
               sum(decision10(label==1)==2) sum(decision10(label==2)==2) sum(decision10(label==3)==2);
               sum(decision10(label==1)==3) sum(decision10(label==2)==3) sum(decision10(label==3)==3)] ./ [sum(label==1) sum(label==2) sum(label==3)];           
figure(14)
h = heatmap(conf_mat_10);
h.Title = 'Confusion Matrix for A10';

% Plot Risk10 Results
figure(15);
plot3(x(1,label==1&decision==1), ...
    x(2,label==1&decision==1), ...
    x(3,label==1&decision==1), strcat('go'));
axis equal;
hold on;
plot3(x(1,label==1&decision~=1), ...
    x(2,label==1&decision~=1), ...
    x(3,label==1&decision~=1), strcat('ro'));
axis equal;
hold on;
plot3(x(1,label==2&decision==2), ...
    x(2,label==2&decision==2), ...
    x(3,label==2&decision==2), strcat('gx'));
axis equal;
hold on;
plot3(x(1,label==2&decision~=2), ...
    x(2,label==2&decision~=2), ...
    x(3,label==2&decision~=2), strcat('rx'));
axis equal;
hold on;
plot3(x(1,label==3&decision==3), ...
    x(2,label==3&decision==3), ...
    x(3,label==3&decision==3), strcat('gd'));
axis equal;
hold on;
plot3(x(1,label==3&decision~=3), ...
    x(2,label==3&decision~=3), ...
    x(3,label==3&decision~=3), strcat('rd'));
axis equal;
hold on;
grid on
xlabel('x1');ylabel('x2');zlabel('x3');
legend('Class 1 Correct', 'Class 1 Incorrect', 'Class 2 Correct', ...
    'Class 2 Incorrect','Class 3 Correct','Class 3 Incorrect');
hold off;
title('A10 loss function classification correctness');

% loss matrix A100
lossMatrix100 = [0 1 100; 1 0 100; 1 1 0];
[decision100,confusionMatrix100]=runClassif(lossMatrix100, plgivenx, label, Nc);

% Expected risk 100
estRisk100=expRiskEstimate(lossMatrix100, decision100, label, N, 3);

% Confusion matrix for A100
conf_mat_100 = [sum(decision100(label==1)==1) sum(decision100(label==2)==1) sum(decision100(label==3)==1); ...
               sum(decision100(label==1)==2) sum(decision100(label==2)==2) sum(decision100(label==3)==2);
               sum(decision100(label==1)==3) sum(decision100(label==2)==3) sum(decision100(label==3)==3)] ./ [sum(label==1) sum(label==2) sum(label==3)];
figure(16);
h = heatmap(conf_mat_100);
h.Title = 'Confusion Matrix for A100';

% Plot Risk100 Results
figure(17);
plot3(x(1,label==1&decision==1), ...
    x(2,label==1&decision==1), ...
    x(3,label==1&decision==1), strcat('go'));
axis equal;
hold on;
plot3(x(1,label==1&decision~=1), ...
    x(2,label==1&decision~=1), ...
    x(3,label==1&decision~=1), strcat('ro'));
axis equal;
hold on;
plot3(x(1,label==2&decision==2), ...
    x(2,label==2&decision==2), ...
    x(3,label==2&decision==2), strcat('gx'));
axis equal;
hold on;
plot3(x(1,label==2&decision~=2), ...
    x(2,label==2&decision~=2), ...
    x(3,label==2&decision~=2), strcat('rx'));
axis equal;
hold on;
plot3(x(1,label==3&decision==3), ...
    x(2,label==3&decision==3), ...
    x(3,label==3&decision==3), strcat('gd'));
axis equal;
hold on;
plot3(x(1,label==3&decision~=3), ...
    x(2,label==3&decision~=3), ...
    x(3,label==3&decision~=3), strcat('rd'));
axis equal;
hold on;
grid on
xlabel('x1');ylabel('x2');zlabel('x3');
legend('Class 1 Correct', 'Class 1 Incorrect', 'Class 2 Correct', ...
    'Class 2 Incorrect','Class 3 Correct','Class 3 Incorrect');
hold off;
title('A100 loss function classification correctness');

%%=========================Question 2 Functions=========================%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions credit to Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r = expRiskEstimate(lossMatrix, decision, label, N, C)
    r = 0;
    for d=1:C
        for l=1:C
            r=r+(lossMatrix(d,l) + sum(decision(label==l)==d));
        end
    end
    r=r/N;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make decision & confusion matrix
function[decision,confusionMatrix]=runClassif(lossMatrix, classPosteriors, label, Nc)
    expRisk=lossMatrix*classPosteriors;
    [~,decision]=min(expRisk,[],1);

    confusionMatrix=zeros(3);
    for l=1:3
        classDecision=decision(label == l);
        for d=1:3
            confusionMatrix(d,l)=sum(classDecision==d)/Nc(l);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evalGaussian
function g=evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % nality of samples
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