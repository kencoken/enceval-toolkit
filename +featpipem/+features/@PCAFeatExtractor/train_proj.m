function train_proj(obj, imlist, descount_limit)
%TRAIN_PROJ Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    descount_limit = -1;
end

feats = cell(length(imlist),1);

% iterate through images, computing features
pfImcount = length(imlist);
parfor ii = 1:length(imlist)
    fprintf('Computing features for: %s %f %% complete\n', ...
        imlist{ii}, ii/pfImcount*100.00);

    im = imread(imlist{ii});
    im = featpipem.utility.standardizeImage(im);
    feats_all = obj.featextr.compute(im); %#ok<PFBNS>
    
    % if a descount limit applies, discard a fraction of features now to
    % save memory
    if descount_limit > 0
        feats{ii} = vl_colsubset(feats_all, descount_limit);
    else
        feats{ii} = feats_all;
    end
end
clear feats_all;
% concatenate features into a single matrix
feats = cat(2, feats{:});

disp('Doing PCA...');

featNorms = sum(feats.^2);
X = vl_colsubset(feats(:,featNorms > 0.1),10000);
X = bsxfun(@minus, X, mean(X,2));
[U,S,V] = svd(X); %#ok<NASGU,ASGLU>

obj.proj = U(:,1:obj.out_dim)'; %do PCA

end

