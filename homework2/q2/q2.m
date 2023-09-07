%% Question 2

%generate data using the given function
nTr = 100;
nVal = 1000;
[xTr,yTr,xVal,yVL] = hw2q2(nTr,nVal);

%get the x1 and x2 training and validation samples
x1Tr = xTr(1,:);
x2Tr = xTr(2,:);
x1Val = xVal(1,:);
x2Val = xVal(2,:);

%cubic vectors for validation and training set
xTrVect = [ones(1,nTr); x1Tr; x2Tr; x1Tr.^2; x1Tr.*x2Tr; x2Tr.^2; x1Tr.^3; x1Tr.^2 .* x2Tr;x1Tr.*x2Tr.^2;x2Tr.^3];
xValVect = [ones(1,nVal); x1Val; x2Val; x1Val.^2; x1Val.*x2Val; x2Val.^2; x1Val.^3; x1Val.^2 .* x2Val;x1Val.*x2Val.^2;x2Val.^3];

%calculate terms derived from textbook
R = (xTrVect * xTrVect')/nTr;
Q = (xTrVect * yTr')/nTr;
wmle = inv(R)*Q;

%calculate mse
MSEmle = mean((yVL - wmle'*xValVect).^2);

%MAP Estimate with gamma
gamma = 10^(-4):.1:10^4;
wmap = zeros(10,length(gamma));
MSEmap=zeros(10,length(gamma));
parameter = zeros(1, length(gamma));
sigma  = 1; 
figure;
for i = 1 : length(gamma)
    parameter(i) = (sigma^2)/(nTr*(10^gamma(i)));
    wmap(:,i) = inv(R+parameter(i)*eye(10))*Q;
    MSEmap(:,i) = mean((yVL-wmap(:,i)'*xValVect).^2);
    plot(gamma(i), MSEmle,'r.', gamma(i), MSEmap(:,i),'b.');
    hold on;
    title('MSE vs Gamma')
    xlabel('log(gamma)'); ylabel('MSE');
    legend('MAP Estimator', 'ML Estimator');

end

function [xTrain,yTrain,xValidate,yValidate] = hw2q2(Ntrain,Nvalidate)
%Ntrain = 100; 
data = generateData(Ntrain);
figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
xTrain = data(1:2,:); yTrain = data(3,:);
%Nvalidate = 1000; 
data = generateData(Nvalidate);
figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
xValidate = data(1:2,:); yValidate = data(3,:);
end
%%
function x = generateData(N)
gmmParameters.priors = [.3,.4,.3]; % priors should be a row vector
gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
[x,labels] = generateDataFromGMM(N,gmmParameters);
end
%%
function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end