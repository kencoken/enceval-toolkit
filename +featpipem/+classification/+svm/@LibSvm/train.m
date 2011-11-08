function train(obj, input, labels)
%TRAIN Testing function for LIBSVM
%   Refer to GenericSVM for interface definition

    % ensure input is of correct form
    input = input';
    if ~issparse(input)
        input = sparse(double(input'));
    end
    
    % prepare temporary output model storage variables
    libsvm = cell(1,classCount);
    libsvm_flipscore = zeros(1,classCount);

    % train models for each class in turn
    parfor ci = 1:length(labels)
        labels_cls = -ones(1,length(labels));
        labels_cls(labels{k}) = 1;
        
        libsvm{ci} = svmtrain(labels_cls, input, ...
            sprintf(' -t 0 -c %f', obj.c)); %#ok<PFBNS>
        % in single class classification, first label encountered is
        % assigned to +1, so if the opposite is true in the label set,
        % set a flag in the libsvm struct to indicate this
        libsvm_flipscore(ci) = (labels_cls(1) == -1);
    end
    
    % copy across trained model
    obj.model = struct;
    obj.model.libsvm = libsvm;
    obj.model.libsvm_flipscore = libsvm_flipscore;
    
    % apply bias multiplier if required
    if obj.bias_multiplier ~= 1
        for i = 1:length(obj.model.libsvm)
            obj.model.libsvm{i}.rho = ...
                obj.bias_multiplier*obj.model.libsvm{i}.rho;
        end
    end
end

