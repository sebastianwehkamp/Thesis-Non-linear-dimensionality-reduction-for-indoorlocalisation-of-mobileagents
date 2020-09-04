%% Reduce set size
cv = cvpartition(size(strengths, 1),'HoldOut',0.95);
idx = cv.test;
data = strengths(~idx,:);
data = zscore(data);
lab = labels(~idx);

%% Training
[LGMLVQ_model,~,trainError] = LGMLVQ_train(data, lab);
dists_local = zeros(size(data,1), size(data,1));
lvq_weights = LGMLVQ_model.w;
lvq_omegas = LGMLVQ_model.psis;

for i = 1:size(data)
    % Find closest LVQ prototype and select corresponding omega
    minDist = inf;
    prototype = 0;  
    x = data(i,:);
    for j = 1:size(lvq_weights,1)
        dist = (x - lvq_weights(j,:))*lvq_omegas{j}'*lvq_omegas{j}*(x - lvq_weights(j,:))';
        if dist<minDist
            minDist = dist;
            prototype = j;
        end
    end
    omega = lvq_omegas{prototype};
    for k = 1:size(data,1)
        y = data(k,:);
        dists_local(i,k) = (x-y)*omega'*omega*(x-y)';
    end
end
dists_local_norm = dists_local/max(abs(dists_local(:)));

dists_mahalanobis = squareform(pdist(data,'Mahalanobis')); 
dists_mahalanobis_norm = dists_mahalanobis/max(abs(dists_mahalanobis(:)));
alpha = 0.04;
dists = zeros(size(data,1), size(data,1));
dists = alpha .* dists_mahalanobis_norm + ((1-alpha) .* dists_local_norm);
Y = sne(dists,2 ,40);
%Y = tsne_d(dists, lab, 2, 30);
gscatter(Y(:,1),Y(:,2),lab);
title(['TSNE with Mahalanobis and local distance using alpha = ' num2str(alpha)]);


