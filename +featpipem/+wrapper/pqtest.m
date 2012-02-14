function res = pqtest(prms, codebook, featextr, encoder, pooler, classifier)
%TEST Summary of this function goes here
%   Detailed explanation goes here

% --------------------------------
% Prepare output filenames
% --------------------------------

trainSetStr = [];
for si = 1:length(prms.splits.train)
    trainSetStr = [trainSetStr prms.splits.train{si}]; %#ok<AGROW>
end

testSetStr = [];
for si = 1:length(prms.splits.test)
    testSetStr = [testSetStr prms.splits.test{si}]; %#ok<AGROW>
end

kChunkIndexFile = fullfile(prms.paths.codes, sprintf('%s_chunkindex.mat', prms.experiment.codes_suffix));
kPQCodebookFile = prms.pqcodebook;
kPQCodesFile = fullfile(prms.paths.codes, sprintf('%s_pqcodes.mat', prms.experiment.codes_suffix));
kPQInvIdxFile = fullfile(prms.paths.compdata, sprintf('%s_pqinvidx.mat', prms.experiment.codes_suffix));
kClassifierFile = fullfile(prms.paths.compdata, sprintf('%s_%s_pqclassifier%s.mat', prms.experiment.name, trainSetStr, prms.experiment.classif_tag));
kResultsFile = fullfile(prms.paths.results, sprintf('%s_%s_pqresults%s.mat', prms.experiment.name, testSetStr, prms.experiment.classif_tag));

% --------------------------------
% Compute Chunks (for all splits)
% --------------------------------
if exist(kChunkIndexFile,'file')
    load(kChunkIndexFile)
else
    chunk_files = featpipem.chunkio.compChunksIMDB(prms, featextr, pooler);
    % save chunk_files to file
    save(kChunkIndexFile, 'chunk_files');
end


train_chunks = cell(1,length(prms.splits.train));
for si = 1:length(prms.splits.train)
    train_chunks{si} = chunk_files(prms.splits.train{si});
end

% --------------------------------
% Compute PQ Codebook
% --------------------------------

if exist(kPQCodebookFile,'file')
    load(kPQCodebookFile);
else
    pqcodebkgen = featpipem.pq.PQCodebkGen(codebook, prms.subquant_count, prms.subquant_bits);
    trainvecs = featpipem.chunkio.loadChunksIntoMat(train_chunks);
    pqcodebook = pqcodebkgen.train(trainvecs);
    save(kPQCodebookFile, 'pqcodebook');
    clear trainvecs pqcodebkgen;
end

% --------------------------------
% Compute PQ Codes
% --------------------------------

pqencoder = featpipem.pq.PQEncoder(pqcodebook);

if exist(kPQCodesFile,'file')
    load(kPQCodesFile);
else
    pqcodes = containers.Map();
    for setName = keys(chunk_files)
        fprintf('Computing PQ codes for set %s...\n',setName{1});
        maxidx = 0;
        chunk_files_set = chunk_files(setName{1});
        for ci = 1:length(chunk_files_set)
            ch = load(chunk_files_set{ci});
            
            if ch.index(end) > maxidx, maxidx = ch.index(end); end
            
            % if this is first chunkfile, preallocate output matrix
            if (ci == 1)
                featcount = size(ch.chunk,2);
                chunkfilecount = length(chunk_files_set);
                pqcodelen = pqencoder.get_output_dim();
                pqcodecls = pqencoder.get_output_class();
                pqcodes_set = cast(zeros(pqcodelen, featcount*chunkfilecount), pqcodecls);
            end
            
            % now compute the pq codes for the current chunk
            chunk_offset = ch.index(1)-1; % compute offset in current chunk
            for codeidx = ch.index
                fprintf('Computing PQ code %d...\n',codeidx);
                pqcodes_set(:, codeidx) = pqencoder.encode(ch.chunk(:, codeidx-chunk_offset));
            end
        end
        
        pqcodes(setName{1}) = pqcodes_set;
        clear pqcodes_set;
        
        % finally, downsize the output matrix if required
        if (maxidx < size(pqcodes(setName{1}),2))
            pqcodestmp = pqcodes(setName{1});
            pqcodes(setName{1}) = pqcodestmp(:,1:maxidx);
            clear pqcodestmp;
        end
    end
    
    % save the codes to file
    save(kPQCodesFile, 'pqcodes');
end

% % --------------------------------
% % Compute PQ Index
% % --------------------------------
% 
% if exist(kPQInvIdxFile, 'file')
%     load(kPQInvIdxFile);
% else
%     pqindexer = featpipem.pq.PQIndex(pqcodebook);
%     for i = 1:length(prms.splits.test)
%         fprintf(['Indexing for ' prms.splits.test{i} ' set...\n']);
%         pqindexer.index(pqcodes(prms.splits.test{i}));
%     end
%     
%     save(kPQInvIdxFile, 'pqindexer');
% end

% % --------------------------------
% % Compute Kernel (if using a dual classifier)
% % --------------------------------
% if isa(classifier, 'featpipem.classification.svm.LibSvmDual')
%     if exist(kKernelFile,'file')
%         load(kKernelFile);
%     else
%         kernelSize = 0;
%         kIdxStart = zeros(length(prms.splits.train),1);
%         kIdxStart(1) = 1;
%         kIdxLen = zeros(length(prms.splits.train),1);
%         for si = 1:length(prms.splits.train)
%             add_dim = size(pqcodes(prms.splits.train{i}),2);
%             kernelSize = kernelSize + add_dim;
%             kIdxLen(si) = add_dim;
%             if si < length(prms.splits.train)
%                 kIdxStart(si+1) = kIdxStart(si) + add_dim;
%             end
%         end
%         
%         K = zeros(kernelSize);
%         
%         for si = 1:length(prms.splits.train)
%             for sj = 1:length(prms.splits.train)
%                 K(kIdxStart(si):kIdxLen(si),kIdxStart(sj):kIdxLen(sj)) = ...
%                     pqcodes(prms.splits.train{si}).*...
%                     pqcodes(prms.splits.train{sj})';
%             end
%         end
% 
%         % save kernel matrix to file
%         save(kKernelFile, 'K');
%     end
% end

% --------------------------------
% Train Classifier
% --------------------------------
if isa(classifier, 'featpipem.classification.svm.LibSvmDual')
    error('No current support for dual');
else
    % ...........................
    % training for svm in primal
    % ...........................
    if exist(kClassifierFile,'file')
        load(kClassifierFile);
        classifier.set_model(model); %#ok<NODEF>
    else
        labels_train = featpipem.utility.getImdbGT(prms.imdb, prms.splits.train, 'concatOutput', true);

        trainvecs_count = 0;
        for si = 1:length(prms.splits.train)
            trainvecs_count = trainvecs_count + size(pqcodes(prms.splits.train{si}),2);
            trainvecs_dim = size(pqcodes(prms.splits.train{si}),1);
        end
        trainvecs = cast(zeros(trainvecs_dim, trainvecs_count), pqencoder.get_output_class());
        startidx = 1;
        for si = 1:length(prms.splits.train)
            endidx = startidx + size(pqcodes(prms.splits.train{si}),2) - 1;
            trainvecs(:,startidx:endidx) = pqcodes(prms.splits.train{si});
            startidx = startidx + size(pqcodes(prms.splits.train{si}),2);
        end

        classifier.train(trainvecs, labels_train);
        model = classifier.get_model(); %#ok<NASGU>
        save(kClassifierFile,'model');
        clear trainvecs;
    end
end

% --------------------------------
% Test Classifier
% --------------------------------
scoremat = cell(1,length(prms.splits.test));
res = cell(1,length(prms.splits.test));
% apply classifier to all testsets in prms.splits.test
for si = 1:length(prms.splits.test)
    [scoremat{si}, scoremat{si}] = classifier.test(pqcodes(prms.splits.test{si}));
    switch prms.experiment.evalusing
        case 'precrec'
            res{si} = featpipem.eval.evalPrecRec(prms.imdb, scoremat{si}, prms.splits.test{si}, prms.experiment.dataset);
        case 'accuracy'
            res{si} = featpipem.eval.evalAccuracy(prms.imdb, scoremat{si}, prms.splits.test{si});
        otherwise
            error('Unknown evaluation method %s', prms.experiment.evalusing);
    end
end

% package results
results.res = res;
results.scoremat = scoremat; %#ok<STRNU>
parameters.prms = prms;
parameters.codebook = codebook;
parameters.pqcodebook = pqcodebook;
parameters.featextr = featextr;
parameters.encoder = encoder;
parameters.pooler = pooler;
parameters.classifier = classifier; %#ok<STRNU>
% save results to file
save(kResultsFile, 'results', 'parameters','-v7.3');
    
end

