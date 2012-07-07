function code = encode(obj, feats)
%ENCODE Encode features using the VLAD method with either hard or soft
%assignment

    % call sub-encoder to get hard/soft assignment
    assign = obj.subencoder.get_assignments(feats);
    
    % word occurence
    word_num = full(sum(assign, 2)');
        
    code = double(feats) * assign';
    
    % subtract cluster centers
    norm_cluster = true;
    
    if norm_cluster
        % normalise the mean vector of each cluster by the cluster size, then subtract the cluster center
        non_empty_clusters = (word_num ~= 0);
        code(:, non_empty_clusters) = bsxfun(@times, code(:, non_empty_clusters), 1 ./ word_num(non_empty_clusters)) - obj.subencoder.codebook_(:, non_empty_clusters);
    else
        % original VLAD (no normalisation, the larger the cluster, the higher the impact
        code = code - bsxfun(@times, obj.subencoder.codebook_, word_num);
    end
    
    % vectorise
    code = code(:);

end

