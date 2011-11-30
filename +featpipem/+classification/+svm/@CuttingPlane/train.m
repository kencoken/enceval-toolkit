function train(obj, input, labels)
%TRAIN Testing function for MATLAB cutting plane method
%   Refer to GenericSVM for interface definition
    
    % calculate lambda parameter
    lambda = 1/(obj.c*size(input,2));
    
    % prepare output model
    w_full = zeros(size(input,1)+1,length(labels));
    
    parfor ci = 1:length(labels)
        labels_cls = -ones(size(input,2),1);
        labels_cls(labels{ci}) = 1;
        
        w_full(:,ci) = linearsvm([input; ones(1,size(input,2))], ...
            labels_cls, lambda, 'maxNumIterations', 20/lambda);
    end
    
    % copy across trained model
    obj.model.w = w_full(1:end-1, :);
    obj.model.b = obj.bias_mul*w_full(end, :);
end

