% parallel
run('/users/karen/src/img_class/enceval-toolkit/startup.m');

DataDir = '/work/karen/img_class/enceval/data_sift_FK/';

ensure_dir(fullfile(DataDir, 'codebooks/'));
ensure_dir(fullfile(DataDir, 'codes/'));
ensure_dir(fullfile(DataDir, 'compdata/'));
ensure_dir(fullfile(DataDir, 'results/'));

% bSIFT = true;
bCrossValSVM = true;

% VocSize = 4000;
VocSize = 256;

%% initialize experiment parameters
prms.experiment.name = 'FKtest'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'FKtest'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load('/work/karen/img_class/enceval/imdb/imdb-VOC2007.mat'); % IMDB file
prms.codebook = fullfile(DataDir, sprintf('codebooks/GMM_%d.mat', VocSize)); % desired location of codebook
prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported
prms.experiment.evalusing = 'precrec'; % evaluation method - currently only precision recall supported

prms.paths.dataset = '/data/pascal/VOCdevkit_2007/'; % path to datasets
prms.paths.codes = fullfile(DataDir,'codes/'); % path where codefiles should be stored
prms.paths.compdata = fullfile(DataDir,'compdata/'); % path where all other compdata (kernel matrices, SVM models etc.) should be stored
prms.paths.results = fullfile(DataDir,'results/'); % path where results should be stored

prms.chunkio.chunk_size = 20; % number of encodings to store in single chunk
prms.chunkio.num_workers = max(matlabpool('size'), 1); % number of workers to use when generating chunks

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

% initialize experiment classes
featextr = featpipem.features.PhowExtractor();
featextr.step = 3;

codebkgen = featpipem.codebkgen.GMMCodebkGen(featextr, VocSize);
codebkgen.descount_limit = 1e6;

%% TRAIN/LOAD CODEBOOK
% -------------------------------------------

codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

% initialize more experiment classes
encoder = featpipem.encoding.FKEncoder(codebook);
encoder.pnorm = single(0.0);

pooler = featpipem.pooling.SPMPooler(encoder);
pooler.subbin_norm_type = 'none';
pooler.norm_type = 'l2';
pooler.pool_type = 'max';
pooler.kermap = 'none';

%% classification
classifier = featpipem.classification.svm.LibSvmDual();

if bCrossValSVM
        
    c = 1:0.2:8;
    
    for ci = 1:length(c)
        
        prms.experiment.classif_tag = sprintf('c%g', c(ci));
        classifier.c = c(ci);
        
        AP{ci} = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
        
    end
    
else
    
    classifier.c = 6.6;
    AP = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);

end
