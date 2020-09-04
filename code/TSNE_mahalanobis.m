%% Reduce set size
cv = cvpartition(size(strengths, 1),'HoldOut',0.95);
idx = cv.test;
data = strengths(~idx,:);
data = zscore(data);
lab = labels(~idx);

% Use matlabs TSNE not toolbox TSNE!
Y = tsne(data,'Algorithm','exact','Distance','mahalanobis', 'perplexity', 50);
gscatter(Y(:,1),Y(:,2),lab);
title('tSNE with Mahalanobis distance');
