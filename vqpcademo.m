% initialize experiment parameters
prms.experiment.name = 'vqpcademo'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'vqpcademo'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(fullfile(pwd,'imdb/imdb-VOC2007.mat')); % IMDB file
prms.codebook = fullfile(pwd,'data/codebooks/phow4k_pca80.mat'); % desired location of codebook
prms.experiment.dataset = 'VOC2007'; % dataset name - currently only VOC2007 supported
prms.experiment.evalusing = 'precrec'; % evaluation method - currently only precision recall supported

prms.paths.dataset = TPDATASETPATH; % path to datasets
prms.paths.codes = fullfile(pwd,'data/codes/'); % path where codefiles should be stored
prms.paths.compdata = fullfile(pwd,'data/compdata/'); % path where all other compdata (kernel matrices, SVM models etc.) should be stored
prms.paths.results = fullfile(pwd,'data/results/'); % path where results should be stored

prms.chunkio.chunk_size = 100; % number of encodings to store in single chunk
prms.chunkio.num_workers = 8; % number of workers to use when generating chunks
% initialize split parameters
prms.splits.train = {'train', 'val'}; % cell array of IMDB splits to use when training
prms.splits.test = {'test'}; % cell array of IMDB splits to use when testing
% initialize experiment classes
featextr_phow = featpipem.features.PhowExtractor();
featextr_phow.step = 3;
featextr = featpipem.features.PCAFeatExtractor(featextr_phow, 80);

%ids = find(prms.imdb.images.set == prms.imdb.sets.TRAIN);
%id_idxs = randperm(length(ids));
%if length(id_idxs) > 200, id_idxs = id_idxs(1:200); end
%imlist = {1,length(id_idxs)};
%for i = 1:length(id_idxs)
%	  imlist{i} = fullfile(prms.paths.dataset, prms.imdb.images.name{ids(id_idxs(i))});
%end
%
%featextr.train_proj(imlist);
%featextr.save_proj(fullfile(pwd,'data/compdata/128to80_proj.mat'));
featextr.load_proj(fullfile(pwd,'data/compdata/128to80_proj.mat'));

codebkgen = featpipem.codebkgen.KmeansCodebkGen(featextr, 4000);
codebkgen.descount_limit = 10e5;

% TRAIN/LOAD CODEBOOK
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
classifier.c = 2.2; % SVM c parameter

% EVALUATE OVER DATASET (returns AP for each class)
% -------------------------------------------
AP = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
