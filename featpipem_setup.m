function featpipem_setup()

cd +featpipem/+lib

archstr = computer('arch');

if(strcmp(archstr,'win64'))
    lapacklib = fullfile(matlabroot, ...
        'extern', 'lib', 'win64', 'microsoft', 'libmwlapack.lib');
    blaslib = fullfile(matlabroot, ...
        'extern', 'lib', 'win64', 'microsoft', 'libmwblas.lib');
    command =  'mex(''LLCEncodeHelper.cpp'', lapacklib, blaslib, largeArrayDims)';
elseif(strcmp(archstr,'win32'))
    lapacklib = fullfile(matlabroot, ...
        'extern', 'lib', 'win32', 'microsoft', 'libmwlapack.lib');
    blaslib = fullfile(matlabroot, ...
        'extern', 'lib', 'win32', 'microsoft', 'libmwblas.lib');
    command =  'mex(''LLCEncodeHelper.cpp'', lapacklib, blaslib)';
elseif strcmp(archstr,'glnx86')
    command = 'mex -O LLCEncodeHelper.cpp -lmwlapack -lmwblas';
elseif strcmp(archstr,'glnxa64')
    command = 'mex -O LLCEncodeHelper.cpp -lmwlapack -lmwblas -largeArrayDims';
else
    error('System architecture could not be identified');
end

mexCmds=cell(0,1);
mexCmds{end+1}=command;

for i=1:length(mexCmds)
  fprintf('Executing %s\n',mexCmds{i});
  eval(mexCmds{i});
end

% cd ../../
% 
% cd +featpipem/+classification/+svm/@LibLinearMex
% 
% mex -g -O manage_class.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% mex -g -O mextest.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% mex -g -O mextrain.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% 
% cd ../@LibLinearTruncMex
% 
% mex -g -O manage_class.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% mex -g -O mextest.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% mex -g -O mextrain.cc -I/home/ken/lib/shared/vlfeat/vl -I/home/ken/lib/shared/liblinear-1.7 -I/home/ken/src/modules/featpipec/src -L/home/ken/src/modules/featpipec/lib -lfeatpipec
% 
% 
% cd ../../../../



