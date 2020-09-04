%net = selforgmap([12 62], 100, 3);
%net = train(net, data');

% Init the grid
msize = [62 12];

% Prepare data
sData = som_data_struct(data);
sData = som_label(sData, 'add', find(lab == 1), 'Aisle 1');
sData = som_label(sData, 'add', find(lab == 2), 'Aisle 2');
sData = som_label(sData, 'add', find(lab == 3), 'Aisle 3');
sData = som_label(sData, 'add', find(lab == 4), 'Aisle 4');
sData = som_label(sData, 'add', find(lab == 5), 'Main aisle');

sMap  = som_randinit(data, 'msize', msize);
sMap  = som_seqtrain(sMap,sData);
sMap = som_autolabel(sMap,sData,'vote');


%% Visualise
som_show(sMap,'umat','all','empty','Labels')
som_show_add('label',sMap,'Textsize',8,'TextColor','r','Subplot',2)


figure;
som_cplane(sMap,som_label2num(sMap));
hold on
som_cplane('hexa',msize,'none')
hold off 
title('Self-organizing map')







