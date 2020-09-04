%% Load training data
trainSet = data;
trainLab = lab;
[dataRow, dataCol] = size(trainSet);

%% Perform LVQ
[trainSet, mu, sig] = zscore(trainSet);

% Training
[LGMLVQ_model,~,trainError] = LGMLVQ_train(trainSet, trainLab);

lvq_weights = LGMLVQ_model.w;
lvq_omegas = LGMLVQ_model.psis;

%% Peform SOM
msize = [65 12];

% Prepare data
sData = som_data_struct(trainSet);
sData = som_label(sData, 'add', find(lab == 1), 'Aisle 1');
sData = som_label(sData, 'add', find(lab == 2), 'Aisle 2');
sData = som_label(sData, 'add', find(lab == 3), 'Aisle 3');
sData = som_label(sData, 'add', find(lab == 4), 'Aisle 4');
sData = som_label(sData, 'add', find(lab == 5), 'Main aisle');

sMap  = som_randinit(sData, 'msize', msize);

%sMap  = som_seqtrain_lvq(sMap,trainSet,lvq_omegas, lvq_weights, 'radius',[5 1],'trainlen',20, 'alpha', 0.001);
sMap  = som_seqtrain_lvq(sMap,trainSet,lvq_omegas, lvq_weights);



sMap = som_autolabel(sMap,sData,'vote');

som_show(sMap,'umat','all','empty','Labels')
som_show_add('label',sMap,'Textsize',8,'TextColor','r','Subplot',2)

figure;
som_cplane(sMap,som_label2num(sMap));
hold on
som_cplane('hexa',msize,'none')
hold off 
title('Supvised SOM results')