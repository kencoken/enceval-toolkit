function train(obj, input, labels)
%TRAIN Testing function for LIBLINEAR
%   Refer to GenericSVM for interface definition

    % ensure input is of correct form
    if ~issparse(input)
        input = sparse(double(input));
    end
    
    % prepare output model
    w_full = zeros(size(input,1)+1,length(labels));
    
    for ci = 1:length(labels)
        labels_cls = -ones(size(input,2),1);
        labels_cls(labels{ci}) = 1;

        svm = train(double(labels_cls), ...
            input,  ...
            sprintf(' -s 3 -B 1 -c %f', obj.c), 'col') ; %#ok<PFBNS>
        w_full(:,ci) = svm.w';
        % in single class classification, first label
        % encountered is assigned to +1, so if the opposite is
        % true in the label set, flip w
        if labels_cls(1) == -1
            w_full(:,ci) = -w_full(:,ci);
        end
    end
    
    % copy across trained model
    obj.model.w = w_full(1:end-1, :);
    obj.model.b = obj.bias_mul*w_full(end, :);
end

