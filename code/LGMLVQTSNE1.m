
%% Configure the train and test set correct
omegas = LGMLVQ_model.psis;
w = LGMLVQ_model.w;
c_w = LGMLVQ_model.c_w;

nb_classes = 5;
nb_dims = 39;
nb_samplesPC = 100;
smallX = zeros(nb_samplesPC*nb_classes,nb_dims);
smallc_X = zeros(nb_samplesPC*nb_classes,1);
dim = 2;

% Create sample
for c=1:nb_classes
    actSampleIdx = ((c-1)*nb_samplesPC+1:c*nb_samplesPC)';
    smallX(actSampleIdx,:) = data(find(lab==c,nb_samplesPC),:);
    smallc_X(actSampleIdx) = ones(nb_samplesPC,1).*c;
end


P=size(smallX,1);
dist = zeros(P,length(c_w));
for i = 1:length(c_w)
    delta = smallX - ones(P,1) * w(i,:);
    delta(isnan(delta)) = 0;
    dist(1:P,i) = sum( ((delta*omegas{i}'*omegas{i}).*delta) ,2 );
end
[~,index] = min(dist,[],2);
responsis = zeros(size(smallX,1),length(c_w));
for i=1:size(smallX,1)
    responsis(i,c_w(index(i))) = 1;
end
eps1 = 0.8;eps2 = 0.1;max_iter = 300;
[map,mapl] = non_lin_map(smallX,omegas,responsis,dim,'max_iter',max_iter,'epsilon',[eps1,eps2],'perplexity',50);

%% View results
y = zeros(size(smallX,1),dim,length(omegas));
for k=1:length(omegas)
    lok = mapl(k,:);
    rk = responsis(:,k);
    y(:,:,k) = rk(:,ones(1,dim)) .* ((smallX*omegas{k}')*map(:,:,k)'+lok(ones(1,size(smallX,1)),:));

end
Y = sum(y,3);

ypd = bsxfun(@plus, sum(Y' .* Y')', bsxfun(@minus, sum(Y' .* Y'), 2 * (Y* Y')));
NNIdx = zeros(1,size(ypd,1));
for k = 1:size(ypd,1)
    [actVal,actIdx] = sort(ypd(k,:));clear dist;
    NNIdx(k) = actIdx(2); 
end
NNError = 1-sum(smallc_X==smallc_X(NNIdx))/length(smallc_X);
markers = {'s','^','v','p','d','o','<','+','*','>'};rand('twister',100);colors = rand(10,3);nb_classes = length(unique(lab));
figure;hold on;arrayfun(@(i) plot(Y(smallc_X==i,1),Y(smallc_X==i,2),markers{i},'MarkerEdgeColor',colors(i,:),'MarkerFaceColor',colors(i,:),'Markersize',3),1:nb_classes);
title([' Local Linear t-SNE mappings']);
legend('1', '2', '3', '4', '5');