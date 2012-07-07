addpath('/users/karen/parallel_2011/');

ToolboxPath = '/users/karen/src/toolboxes/';

oldDir = cd([ToolboxPath 'vlfeat/toolbox/']);
vl_setup();
cd(oldDir);

% repeatability computation
addpath([ToolboxPath 'libsvm-3.11/matlab/']);

addpath('/users/karen/src/common/');

% learnt descriptor 
addpath('/users/karen/src/learn_desc_class/comp_desc/');

% FK
addpath('/users/karen/src/img_class/gmm-fisher/matlab/');

clear oldDir;
clear ToolboxPath;