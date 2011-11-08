function encoding = KCBEncode(imwords, ...
    dictwords, K, sigma, kdtree, maxComparisons, kcbType, outputFullMat)
%KCBENCODE Summary of this function goes here
%   Detailed explanation goes here

    % START validate input parameters -------------------------------------
    default('K', 5);
    default('sigma', 1e-4);
    % only used if a kdtree is specified
    default('maxComparisons', size(dictwords,2));
    % possible values: 'unc', 'pla', 'kcb'
    % see van Gemert et al. ECCV 2008 for details
    % 'inveuc' simply weights by the inverse of euclidean distance, then l1
    % normalizes
    default('kcbType', 'unc');
    default('outputFullMat', true);
    % END validate input parameters ---------------------------------------
    
    % if using the 'plausibility' method, only 1NN is required
    if strcmp(kcbType, 'pla')
        K = 1;
    end
    
    % -- find K nearest neighbours in dictwords of imwords --
    if (nargin < 5) || isempty(kdtree)
        distsq = vl_alldist2(double(dictwords),double(imwords));
        % distances is MxN matrix where M is num of codewords
        % and N is number of descriptors in imwords
        [distsq, ix] = sort(distsq);
        % ix is a KxN matrix containing
        % the indices of the K nearest neighbours of each image descriptor
        ix(K+1:end,:) = [];
        distsq(K+1:end,:) = [];
    else
        [ix, distsq] = vl_kdtreequery(kdtree, single(dictwords), ...
                                            single(imwords), ...
                                            'MaxComparisons', ...
                                            maxComparisons, ...
                                            'NumNeighbors', K);
        ix = double(ix);
    end
    
    if outputFullMat
        % encoding is MxN sparse matrix of results
        % (where M is number of dictionary words, and N is number of imwords)
        encoding_sr = zeros(K*size(imwords,2),1);
        encoding_ir = zeros(K*size(imwords,2),1);
        encoding_jc = zeros(K*size(imwords,2),1);
        sparseidx = 1;
        sparseidxend = sparseidx + size(ix,1) - 1;
    else
        encoding = zeros(size(dictwords,2),1);
    end
    

    kerMultiplier = 1/(sqrt(2*pi)*sigma);
    kerIntMultiplier = -0.5/(sigma^2);

    % kerDists is a KxN matrix containing the kernel distances of the K
    % nearest neighbours of each image descriptor
    kerDists = kerMultiplier*exp(kerIntMultiplier*distsq);

    switch kcbType
        case {'pla', 'kcb'}
            for idx = 1:size(imwords,2)
                if outputFullMat
                    encoding_sr(sparseidx:sparseidxend) = ...
                        kerDists(:,idx);
                    encoding_ir(sparseidx:sparseidxend) = ...
                        ix(:,idx);
                    encoding_jc(sparseidx:sparseidxend) = ...
                        repmat(idx,size(ix,1),1);
                    sparseidx = sparseidxend + 1;
                    sparseidxend = sparseidx + size(ix,1) - 1;
                else
                    encoding(ix(:,idx)) = encoding(ix(:,idx)) + ...
                        kerDists(:,idx);
                end
            end
        case 'unc'
            for idx = 1:size(imwords,2)
                if nnz(kerDists(:,idx)) == 0
                    warning('FEATPIPE:kcbsigsmall',['The current ' ...
                                        'feature code was calculated ' ...
                                        'as all zeros, which may ' ...
                                        'indicate that KCB sigma ' ...
                                        'is set too small']);
                end
                if outputFullMat
                    encoding_sr(sparseidx:sparseidxend) = ...
                        (kerDists(:,idx)/norm(kerDists(:,idx),1));
                    encoding_ir(sparseidx:sparseidxend) = ...
                        ix(:,idx);
                    encoding_jc(sparseidx:sparseidxend) = ...
                        repmat(idx,size(ix,1),1);
                    sparseidx = sparseidxend + 1;
                    sparseidxend = sparseidx + size(ix,1) - 1;
                else
                    encoding(ix(:,idx)) = encoding(ix(:,idx)) + ...
                        (kerDists(:,idx)/norm(kerDists(:,idx),1));
                end
            end
    end
        
    
    if outputFullMat
        % construct sparse encoding matrix
        encoding = sparse(encoding_ir, encoding_jc, encoding_sr, ...
            size(dictwords,2), size(imwords,2));
    end
end

