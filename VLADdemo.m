% parallel
run('/users/karen/src/img_class/enceval-toolkit/startup.m');

DataDir = '/work/karen/img_class/enceval/data_VLAD_hard_cluster_norm_nz_idf/';

ensure_dir(fullfile(DataDir,'codebooks/'));
ensure_dir(fullfile(DataDir, 'dim_red/'));
ensure_dir(fullfile(DataDir,'codes/'));
ensure_dir(fullfile(DataDir,'compdata/'));
ensure_dir(fullfile(DataDir,'results/'));

bCrossValSVM = true;

VocSize = 512;

% descriptor dimensionality after PCA
desc_dim = 80;

hard_assign = true;

%% initialize experiment parameters
prms.experiment.name = 'VLAD'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'VLAD'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load('/work/karen/img_class/enceval/imdb/imdb-VOC2007.mat'); % IMDB file
prms.codebook = fullfile(DataDir, sprintf('codebooks/kmeans_%d.mat', VocSize)); % desired location of codebook
prms.dimred = fullfile(DataDir, sprintf('dim_red/PCA_%d.mat', desc_dim)); % desired location of low-dim projection matrix
prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported
prms.experiment.evalusing = 'precrec'; % evaluation method - currently only precision recall supported

prms.paths.dataset = '/data/pascal/VOCdevkit_2007/'; % path to datasets
prms.paths.codes = fullfile(DataDir,'codes/'); % path where codefiles should be stored
prms.paths.compdata = fullfile(DataDir,'compdata/'); % path where all other compdata (kernel matrices, SVM models etc.) should be stored
prms.paths.results = fullfile(DataDir,'results/'); % path where results should be stored

prms.chunkio.chunk_size = 40; % number of encodings to store in single chunk
prms.chunkio.num_workers = max(matlabpool('size'), 1); % number of workers to use when generating chunks

% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing

% initialize experiment classes
featextr = featpipem.features.PhowExtractor();
featextr.remove_zero = true;
featextr.step = 3;

%% train/load dimensionality reduction
if desc_dim ~= 128
    dimred = featpipem.dim_red.PCADimRed(featextr, desc_dim);
    featextr.low_proj = featpipem.wrapper.loaddimred(dimred, prms);
else
    % no dimensionality reduction
    featextr.low_proj = [];
end

%% train/load codebook
codebkgen = featpipem.codebkgen.KmeansCodebkGen(featextr, VocSize);
[codebook, word_freq] = featpipem.wrapper.loadcodebook(codebkgen, prms);

%% initialize encoder + pooler
if hard_assign
    % hard assignment
    subencoder = featpipem.encoding.VQEncoder(codebook);
    subencoder.max_comps = -1;
else
    % soft assignment
    subencoder = featpipem.encoding.KCBEncoder(codebook);
    subencoder.max_comps = -1;
    
    % 25 too small
    subencoder.sigma = 50;    
    subencoder.num_nn = 5;
end

encoder = featpipem.encoding.VLADEncoder(subencoder);
encoder.word_freq = word_freq;

pooler = featpipem.pooling.SPMPooler(encoder);
pooler.subbin_norm_type = 'l2';
pooler.norm_type = 'none';
pooler.kermap = 'hellinger';
pooler.post_norm_type = 'l2';
pooler.pool_type = 'sum';

%% classification
classifier = featpipem.classification.svm.LibSvmDual();

if bCrossValSVM
        
    c = 1:0.5:5;
    
    for ci = 1:length(c)
        
        prms.experiment.classif_tag = sprintf('c%g', c(ci));
        classifier.c = c(ci);
        
        AP{ci} = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
        
    end
    
else
    
    classifier.c = 1.6;
    AP = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);

end
