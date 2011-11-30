function train(obj, input, labels)
%TRAIN Testing function for PEGASOS from vl_feat
%   Refer to GenericSVM for interface definition
    
    % calculate lambda parameter
    lambda = 1/(obj.c*size(input,2));
    
    % prepare output model
    w_full = zeros(size(input,1)+1,length(labels));
    
    bias_mul = obj.bias_mul;
    
    parfor ci = 1:length(labels)
        perm = randperm(size(input,2));
        
        labels_cls = -ones(size(input,2),1);
        labels_cls(labels{ci}) = 1;
        
        w_full(:,ci) = vl_pegasos(input(:,perm), ...
            int8(labels_cls(perm)), lambda, 'NumIterations', 20/lambda, ...
            'BiasMultiplier', bias_mul);
    end
    
    % copy across trained model
    obj.model.w = w_full(1:end-1, :);
    obj.model.b = obj.bias_mul*w_full(end, :);
end

