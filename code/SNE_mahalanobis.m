%% Reduce set size
cv = cvpartition(size(strengths, 1),'HoldOut',0.95);
idx = cv.test;
data = strengths(~idx,:);
data = zscore(data);
lab = labels(~idx);


D = squareform(pdist(data,'Mahalanobis')); 
Y = sne(D, 2, 50);
gscatter(Y(:,1),Y(:,2),lab);
title('SNE with Mahalanobis distance');