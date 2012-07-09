% initialize experiment parameters
prms.experiment.name = 'fkpcademo'; % experiment name - prefixed to all output files other than codes
prms.experiment.codes_suffix = 'fkpcademo'; % string prefixed to codefiles (to allow sharing of codes between multiple experiments)
prms.experiment.classif_tag = ''; % additional string added at end of classifier and results files (useful for runs with different classifier parameters)
prms.imdb = load(fullfile(pwd,'imdb/imdb-VOC2007.mat')); % IMDB file
prms.codebook = fullfile(pwd,'data/codebooks/gmm_phow4k_pca80.mat'); % desired location of codebook
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

codebkgen = featpipem.codebkgen.GmmCodebkGen(featextr, 256);
codebkgen.descount_limit = 10e5;

% TRAIN/LOAD CODEBOOK
% -------------------------------------------
codebook = featpipem.wrapper.loadcodebook(codebkgen, prms);

% initialize more experiment classes
encoder = featpipem.encoding.FKEncoder(codebook);
obj.pnorm = single(0.0);
pooler = featpipem.pooling.SPMPooler(encoder);
pooler.subbin_norm_type = 'none';
pooler.norm_type = 'l2';
pooler.pool_type = 'max';
pooler.kermap = 'none';
classifier = featpipem.classification.svm.LibSvmDual();
classifier.c = 6.6;

% EVALUATE OVER DATASET (returns AP for each class)
% -------------------------------------------
% -------------------------------------------
c = [1.6 1.8 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4 4.2 4.4 4.6 4.8 5 5.2 5.4 5.6 5.8 6 6.2 6.4 6.6 6.8 7 7.2 7.4 7.6 7.8];
for ci = 1:length(c)
    prms.experiment.classif_tag = sprintf('c%f', c(ci));
    classifier.c = c(ci);
    AP = featpipem.wrapper.dstest(prms, codebook, featextr, encoder, pooler, classifier);
end
