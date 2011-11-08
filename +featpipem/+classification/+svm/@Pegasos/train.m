function train(obj, input, labels)
%TRAIN Testing function for PEGASOS from vl_feat
%   Refer to GenericSVM for interface definition
    
    % calculate lambda parameter
    lambda = 1/(c*size(input,2));
    
    % prepare output model
    w_full = zeros(size(input,1)+1,length(labels));
    
    parfor ci = 1:length(labels)
        perm = randperm(size(input,2));
        
        labels_cls = -ones(1,length(labels));
        labels_cls(labels{k}) = 1;
        
        w_full(:,ci) = vl_pegasos(input(:,perm), ...
            int8(labels_cls(perm)), lambda, 'NumIterations', 20/lambda, ...
            'BiasMultiplier', biasMultiplier);
    end
    
    % copy across trained model
    obj.model.w = w_full(1:end-1, :);
    obj.model.b = obj.bias_multiplier*w_full(end, :);
end

