function AP = evalPrecRec(imdb, dataset, scoremat, testset)
%EVALPRECREC Summary of this function goes here
%   Detailed explanation goes here

AP = zeros(1,size(scoremat,1));

gt = featpipem.utility.getImdbGT(imdb, {testset}, 'outputSignedLabels', true);

for ci = 1:size(scoremat,1)    
    % get ground truth
    gt_cls = gt{ci};
    if length(gt_cls) ~= size(scoremat,2)
        error('Mismatch between size of scoremat and number of images in testset(s)');
    end
    
    % get indices of current class sorted in descending order of confidence
    [sortidx sortidx] = sort(scoremat(ci,:),'descend'); %#ok<ASGLU>
    tp = gt_cls(sortidx)>0;
    fp = gt_cls(sortidx)<0;
    
    fp = cumsum(fp);
    tp = cumsum(tp);
    rec = tp/sum(gt_cls>0);
    prec = tp./(fp+tp);
    
    AP(ci) = featpipem.eval.VOCdevkit.VOCap(dataset, rec, prec);
    
end

end

