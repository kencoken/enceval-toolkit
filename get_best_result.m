%%
% clear;

ResDir = '/work/karen/img_class/enceval/data_VLAD_hard_cluster_norm/results/';

ResFiles = dir([ResDir '/*.mat']);

nResFiles = numel(ResFiles);

FileNames = cell(nResFiles, 1);
mAP = zeros(nResFiles, 1);

for iFile = 1:nResFiles
    FileNames{iFile} = ResFiles(iFile).name;
    
    load([ResDir FileNames{iFile}], 'results');
    
    if isstruct(results.res{1})
        mAP(iFile) = mean([results.res{1}.AP]);
    else
        mAP(iFile) = mean(results.res{1});
    end
end

%%
[MaxAP MaxAPIdx] = max(mAP);

fprintf('Dir: %s\n', ResDir);
fprintf('MaxAP: %g, file name: %s\n', MaxAP, FileNames{MaxAPIdx});