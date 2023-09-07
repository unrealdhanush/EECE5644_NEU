clearvars; close all; clear;

% Initial parameters
lambda=0.00001; %for regularization

% White Wine Quality Dataset
wine_raw_data = dlmread('winequality-white.csv',';',1,0)';

%Separate dataset into data and true class labels
x_W=wine_raw_data(1:end-1,:);
label_W=wine_raw_data(end,:);

% HAR Dataset (true class labels are separate files so no need to separate)
load X_train.txt;
load X_test.txt;
x_H=vertcat(X_train,X_test)';

%load true class labels
load y_train.txt;
load y_test.txt;
label_H=vertcat(y_train, y_test)';

% Dimensions and size of datasets from matrices
[n_W, N_W]=size(x_W);
[n_H,N_H]=size(x_H);

% Class labels and number of classes
class_W=unique(label_W);
C_W=length(class_W);
class_H=unique(label_H);
C_H=length(class_H);

% Class priors
priors_W=zeros(1,C_W);
for l=1:C_W
    priors_W(l)=sum(label_W==class_W(l))/N_W;
end
priors_H=zeros(1,C_H);
for l=1:C_H
    priors_H(l)=sum(label_H == class_H(l))/N_H;
end

display(priors_W);
display(priors_H);

% Estimated mean vectors
mu_W=zeros(n_W, C_W);
for l=1:C_W
    samples=x_W(:,label_W == class_W(l));
    mu_W(:,l)=mean(samples, 2);
end
mu_H=zeros(n_H, C_H);
for l=1:C_H
    samples=x_H(:,label_H == class_H(l));
    mu_H(:,l)=mean(samples, 2);
end

display(mu_W);
display(mu_H);

% Estimated covariance matrices
Sigma_W=zeros(n_W, n_W, C_W);
for l=1:C_W
    Sigma_W(:,:,l)=cov(x_W(:,label_W==class_W(l))')+(lambda*eye(n_W));
end
Sigma_H = zeros(n_H, n_H, C_H);
for l=1:C_H
    Sigma_H(:,:,l)=cov(x_H(:,label_H==class_H(l))')+(lambda*eye(n_H));
end

display(Sigma_W);
display(Sigma_H);
%[QW,DW]=eig(Sigma_W(:,:,1));
%[QH,DH]=eig(Sigma_H(:,:,1));

% Class posteriors and loss matrices
classPosterior_W=classPosterior(x_W, mu_W, Sigma_W, N_W, C_W, priors_W);
classPosterior_H=classPosterior_Mvnpdf(x_H, mu_H, N_H, C_H, priors_H);

lossMatrix_W=zeros(C_W);
 for i=1:C_W
    for j=1:C_W
        lossMatrix_W(i,j) = abs(i-j);
    end
 end
lossMatrix_H=ones(C_H,C_H)-eye(C_H);
        
% Run classification for confusion matrices and create pError
[decisions_W,confusionMatrix_W]=runClassif(lossMatrix_W, classPosterior_W, label_W, class_W, priors_W);
[decisions_H,confusionMatrix_H]=runClassif(lossMatrix_H, classPosterior_H, label_H, class_H, priors_H);
pError_W=calculatePErr(confusionMatrix_W, priors_W);
pError_H=calculatePErr(confusionMatrix_H, priors_H);
y_W=LDA(x_W',label_W)';
y_H=LDA(x_H',label_H)';
 
%Plot first 3 dimensions of LDAprojections on each dataset
figure(1);
for l=1:C_W
    scatter3(y_W(1, label_W==class_W(l)), y_W(2, label_W==class_W(l)), y_W(3, label_W==class_W(l)), '.');
    hold on;
    axis equal;
end
title('White Wine Dataset Fisher LDA - 3 Dimensions');
xlabel('y1'); ylabel('y2'); zlabel('y3');
legend('Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9');

figure(2);
for l=1:C_H
    scatter3(y_H(1, label_H==class_H(l)), y_H(2, label_H==class_H(l)), y_H(3, label_H==class_H(l)), '.');
    hold on;
    axis equal;
end
title('HAR Dataset Fisher LDA - 3 Dimensions');
xlabel('y1'); ylabel('y2'); zlabel('y3');
legend('Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying')

%Confusion Matrices
figure(3);
h1=heatmap(confusionMatrix_W);
h1.Title='Confusion Matrix for Wine Dataset';

figure(4);
h2=heatmap(confusionMatrix_H);
h2.Title='Confusion Matrix for HAR Dataset';

%given a confusion matrix and corresponding class priors calculate
%probability of error for the classifier
function pErr=calculatePErr(confusionMatrix, prior)
    C=length(prior);
    pErr=0;
    for l=1:C
        for d=1:C
            if d~=l
                pErr=pErr+confusionMatrix(d,l) * prior(l);
            end
        end
    end
end

% Class posterior for samples
function p=classPosterior(x, mu, Sigma, N, C, priors)
    for l=1:C
        pxgivenl(l,:)=evalGaussian(x,mu(:,l),Sigma(:,:,l)');
    end
    px=priors*pxgivenl;
    p=pxgivenl.*repmat(priors',1,N)./repmat(px,C,1);
end

% Class posterior for HAR dataset using mvnpdf and omitting sigma, since the large covariance
% matrices cause issues
function p=classPosterior_Mvnpdf(x, mu, N, C, priors)
    for l=1:C
        pxgivenl(l,:)=mvnpdf(x',mu(:,l)');
    end
    px=priors*pxgivenl;
    p=pxgivenl.*repmat(priors',1,N)./repmat(px,C,1);
end

% Make decisions and confusion matrix
function[decision, confusionMatrix]=runClassif(lossMatrix, classPosterior, label, class, labelCount)
    C=length(class);
    expRisk=lossMatrix*classPosterior;
    [~,decisionInds]=min(expRisk,[],1);
    decision=class(decisionInds);

    confusionMatrix=zeros(C);
    for l=1:C
        classDecision=decision(label==class(l));
        for d=1:C
            confusionMatrix(d,l)=sum(classDecision==d)/labelCount(l);
        end
    end
end

% evalGaussian
% credit to Prof. Deniz's shared code file
function g=evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end

% LDA
% credit online
function Y = LDA(X,L)
    Classes=unique(L)';
    k=numel(Classes);
    n=zeros(k,1);
    C=cell(k,1);
    M=mean(X);
    S=cell(k,1);
    Sw=0;
    Sb=0;
    for j=1:k
        Xj=X(L==Classes(j),:);
        n(j)=size(Xj,1);
        C{j}=mean(Xj);
        S{j}=0;
        for i=1:n(j)
            S{j}=S{j}+(Xj(i,:)-(C{j})'*(Xj(i,:)-C{j}));
        end
        Sw=Sw+S{j};
        Sb=Sb+n(j)*(C{j}-M)'*(C{j}-M);
    end
    [W, LAMBDA]=eig(Sb,Sw);
    lambda=diag(LAMBDA);
    [~, SortOrder]=sort(lambda,'descend');
    W=W(:,SortOrder);
    Y=X*W;
end