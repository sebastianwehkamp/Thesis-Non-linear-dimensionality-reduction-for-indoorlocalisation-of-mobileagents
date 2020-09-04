addpath(genpath('./SOM_Toolbox/'));

% Trainset
trainSet = data;

% Normalize the data to have zero mean and unit variance.
[trainSet, mu, sig] = zscore(trainSet);
trainLab = lab;
train_pos = points;

%% Semi supervised SOM
% Init the grid
msize = [62 12];

% Prepare data
sData = som_data_struct(trainSet);
sData = som_label(sData, 'add', find(lab == 1), 'Aisle 1');
sData = som_label(sData, 'add', find(lab == 2), 'Aisle 2');
sData = som_label(sData, 'add', find(lab == 3), 'Aisle 3');
sData = som_label(sData, 'add', find(lab == 4), 'Aisle 4');
sData = som_label(sData, 'add', find(lab == 5), 'Main aisle');

sMap  = som_randinit(trainSet, 'msize', msize);
warehouse_width = 10;
warehouse_length = 62;

% Initialize P for semi supervised learning
P = initP(msize, warehouse_width, warehouse_length, wap_locs);
P = reshape(P,[numel(P), 1]);
P = bsxfun(@rdivide,bsxfun(@minus,cell2mat(P),mu),sig);

sMap  = semi_som_seqtrain(sMap,trainSet, P);

%sMap  = som_seqtrain(sMap,trainSet, 'radius',[5 1],'trainlen',300, 'alpha', 0.0001);

sMap = som_autolabel(sMap,sData,'vote');


%% Visualise
som_show(sMap,'umat','all','empty','Labels')
som_show_add('label',sMap,'Textsize',8,'TextColor','r','Subplot',2)

figure;
som_cplane(sMap,som_label2num(sMap));
hold on
som_cplane('hexa',msize,'none')
hold off 
title('SOM with 0.1 P known')









