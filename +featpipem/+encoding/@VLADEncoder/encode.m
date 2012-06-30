function code = encode(obj, feats)
%ENCODE Encode features using the VLAD method with either hard or soft
%assignment

    % call sub-encoder to get hard/soft assignment
    assign = obj.subencoder.get_assignments(feats);
    
    % word occurence
    word_num = full(sum(assign, 2)');
        
    code = double(feats) * assign';
    
    % subtract words    
    if true
        code = code - bsxfun(@times, obj.subencoder.codebook_, word_num);
    end
    
    if false
        code = bsxfun(@times, code, 1 ./ word_num) - obj.subencoder.codebook_;
    end
    
    % vectorise
    code = code(:);

end

