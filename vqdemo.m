% parallel
run('/users/karen/src/img_class/enceval-toolkit/startup.m');

DataDir = '/work/karen/img_class/enceval/data_sift/';

ensure_dir(fullfile(DataDir,'codebooks/'));
ensure_dir(fullfile(DataDir,'codes/'));
ensure_dir(fullfile(DataDir,'compdata/'));
ensure_dir(fullfile(DataDir,'results/'));

bSIFT = true;
bCrossValSVM = false;

% VocSize = 4000;
VocSize = 256;

%% initialize experiment parameters
prms.experiment.name = 'vqdemo'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'vqdemo'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(fullfile(pwd,'imdb/imdb-VOC2007.mat')); % IMDB file
prms.codebook = fullfile(DataDir, sprintf('codebooks/%d.mat', VocSize)); % desired location of codebook
prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported
prms.experiment.evalusing = 'precrec'; % evaluation method - currently only precision recall supported

prms.paths.dataset = '/data/pascal/VOCdevkit_2007/'; % path to datasets
prms.paths.codes = fullfile(DataDir,'codes/'); % path where codefiles should be stored
prms.paths.compdata = fullfile(DataDir,'compdata/'); % path where all other compdata (kernel matrices, SVM models etc.) should be stored
prms.paths.results = fullfile(DataDir,'results/'); % path where results should be stored

prms.chunkio.chunk_size = 40; % number of encodings to store in single chunk
prms.chunkio.num_workers = max(matlabpool('size'), 1); % number of workers to use when generating chunks

% prms.chunkio.chunk_size = 100; % number of encodings to store in single chunk
% prms.chunkio.num_workers = 8; % number of workers to use when generating chunks

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

% initialize experiment classes
if bSIFT
    featextr = featpipem.features.PhowExtractor();
    DefaultC = 2.2;
else
    featextr = featpipem.features.LearntExtractor();
    DefaultC = 2.8;
%     DefaultC = 100;
end

featextr.step = 3;

codebkgen = featpipem.codebkgen.KmeansCodebkGen(featextr, VocSize);
codebkgen.descount_limit = 10e5;

%% TRAIN/LOAD CODEBOOK
% -------------------------------------------
codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

% initialize more experiment classes
encoder = featpipem.encoding.VQEncoder(codebook);
encoder.max_comps = 25; % max comparisons used when finding NN using kdtrees
encoder.norm_type = 'none'; % normalization to be applied to encoding (either 'l1' or 'l2' or 'none')

pooler = featpipem.pooling.SPMPooler(encoder);
pooler.subbin_norm_type = 'none'; % normalization to be applied to SPM subbins ('l1' or 'l2' or 'none')
pooler.norm_type = 'l1'; % normalization to be applied to whole SPM vector
pooler.pool_type = 'sum'; % SPM pooling type (either 'sum' or 'max')
pooler.kermap = 'homker'; % additive kernel map to be applied to SPM (either 'none' or 'homker')

classifier = featpipem.classification.svm.LibSvmDual();

%% EVALUATE OVER DATASET (returns AP for each class)
% -------------------------------------------

if bCrossValSVM
    prms.splits.train = {'train'};
    prms.splits.test = {'val'};
    
    c = 1:0.2:8;
    
    for ci = 1:length(c)
        
        prms.experiment.classif_tag = sprintf('c%f', c(ci));
        classifier.c = c(ci);
        AP{ci} = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
        
    end
    
    nTests = numel(AP);
    MeanAP = zeros(nTests, 1);
    
    for i = 1:nTests
        MeanAP(i) = mean(AP{i}{1});
    end
    
    [MaxMeanAP, MaxMeanAPIdx] = max(MeanAP);
    
    fprintf('MaxMeanAP = %g @ C = %g\n', MaxMeanAP, c(MaxMeanAPIdx));
    
else
    classifier.c = DefaultC;
    
    AP = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
end

