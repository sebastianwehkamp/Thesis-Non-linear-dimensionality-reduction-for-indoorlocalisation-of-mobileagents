addpath(genpath('./LVQ_toolbox/'));

%% Configure the train and test set correct
cv = cvpartition(size(strengths, 1),'HoldOut',0.3);
idx = cv.test;

% Trainset
trainSet = strengths(~idx,:);

% Normalize the data to have zero mean and unit variance.
[trainSet, mu, sig] = zscore(trainSet);
trainLab = labels(~idx);

testSet = strengths(idx,:);
test_pos = points(idx,:);
% normalize the data to have zero mean and unit variance.
testSet = bsxfun(@rdivide,bsxfun(@minus,testSet,mu),sig);
testLab = labels(idx);

%% Training
[LGMLVQ_model,~,trainError] = LGMLVQ_train(trainSet, trainLab);

estimatedTrainLabels = LGMLVQ_classify(trainSet, LGMLVQ_model);
trainError = mean( trainLab ~= estimatedTrainLabels );
fprintf('LGMLVQ: error on the train set: %f\n',trainError);

%% Classifaction
estimatedTestLabels = LGMLVQ_classify(testSet, LGMLVQ_model);
testError = mean( testLab ~= estimatedTestLabels );
fprintf('LGMLVQ: error on the test set: %f\n',testError);

%% Visualise
figure,
p0 = scatter(test_pos(:,1), test_pos(:,2),'.','MarkerEdgeColor',[0.9 0.9 0.9]);hold on,
p1 = scatter(test_pos((testLab ~= estimatedTestLabels),1), test_pos((testLab ~= estimatedTestLabels),2),'k.');

title('LGMLVQ aisle evaluation using random initialisation','FontSize',14),
legend(p1,{'Aisle errors'}),
hold off;