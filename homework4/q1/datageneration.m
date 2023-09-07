%% Initialize
clear; close all;
f{1,1} = "3096_color.jpg";
f{1,2} = "67079.jpg";
K = 10;
M = 5;
n = size(f, 2);
%%
for i = 1:n
    imdata  = imread(f{1,i});
    figure(1), subplot(n, 2, i*2-1),
    imshow(imdata);
    title("shows the original photo"); hold on;
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = (1:R)'*ones(1,C); colIndices = ones(R,1)*(1:C);
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column
indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N));
    d = size(x,1);
    model = 2;
    gm = fitgmdist(x',model);
    p = posterior(gm, x');
    [~, l] = max(p,[], 2);
    li = reshape(l, R, C);
    figure(1), subplot(n, 2, i*2)
    imshow(uint8(li*255/model));
    title(strcat("Clustering with K=", num2str(model)));
    ab = zeros(1,M);
 for model = 1:M
        ab(1,model) = calcLikelihood(x, model, K);
end
    [~, mini] = min(ab);
    gm = fitgmdist(x', mini);
    p = posterior(gm, x');
    [~, l] = max(p,[], 2);
    li = reshape(l, R, C);
    figure(2), subplot(n,1,i),
    imshow(uint8(li*255/mini));
    title(strcat("Best Clustering with K=", num2str(mini)));
    fig = figure(3);
    subplot(1,n,i), plot(ab,'-b');
    title(strcat("For image-", num2str(i)));
end
han = axes(fig, 'visible', 'off');
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Negative Loglikelihood');
xlabel(han,'Model Order');
title(han,'Negative loglikelihood vs Model Order');
%% function
function negativeLoglikelihood = calcLikelihood(x, model, K)
    N = size(x,2);
    dummy = ceil(linspace(0, N, K+1));
    negativeLoglikelihood = 0;
    for k=1:K
        indPartitionLimits(k,:) = [dummy(k) + 1, dummy(k+1)];
end
for k = 1:K
        indValidate = (indPartitionLimits(k,1):indPartitionLimits(k,2));
        xv = x(:, indValidate); % Using folk k as validation set
        if k == 1
            indTrain = (indPartitionLimits(k,2)+1:N);
        elseif k == K
            indTrain = (1:indPartitionLimits(k,1)-1);
else
            indTrain = (indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1);
end
        xt = x(:, indTrain);
try
    gm = fitgmdist(xt', model);
    [~, nlogl] = posterior(gm, xv');
    negativeLoglikelihood = negativeLoglikelihood + nlogl;
    catch exception
end 
end
end