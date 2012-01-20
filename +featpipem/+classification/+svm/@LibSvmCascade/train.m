function train(obj, input, labels)
%TRAIN Testing function for LIBSVMCASCADE
%   Refer to GenericSVM for interface definition
    
    % input is of dimensions feat_dim x feat_count
    % i.e. column features

    % convenience variables
    num_classes = length(labels);
    %feat_dim = size(input,1);
    feat_count = size(input,2);

    % ensure input is of correct form
    if ~issparse(input)
        input = sparse(double(input));
    end
    
    labels_cls = -ones(feat_count,num_classes);
    for ci = 1:num_classes
       labels_cls(labels{ci},ci) = 1;
    end
    
    % randomly permute inputs and labels
    permutation = randperm(feat_count);
    input = input(:,permutation);
    for ci = 1:num_classes
        for i = 1:length(labels{ci})
            labels{ci}(i) = permutation(i);
        end
    end
    
    
    % deal with levels obj.num_workers -> 2 of the cascade
    % dividing the number of workers by two each iteration
    worker_count = obj.num_workers;
    c = obj.c;
    first_iter = true;
    while (worker_count > 1)
        % on first run prepare SVsin as a cell array with worker_count
        % elements, each containing class_count vectors. All cells in it
        % are just indices of ALL vectors
        if first_iter
            SVsin = cell(worker_count,1);
            for wi = 1:worker_count
                for ci = 1:num_classes
                    SVsin{wi}{ci} = wi:worker_count:feat_count;
                end
            end
            first_iter = false;
        else
            % precombine SVsin
            SVsin = cell(worker_count,1);
            for wi = 1:worker_count
                for ci = 1:num_classes
                    SVsin{wi}{ci} = [SVsout{wi}{ci}; SVsout{wi+1}{ci}];
                    fprintf('Level: %d, Worker: %d, Class: %d, SVs: %d\n',worker_count,wi,ci,length(SVsin{wi}{ci}));
                end
            end
        end
        
        % preallocate output cell for support vectors
        SVsout = cell(worker_count,1);
        for wi = 1:length(SVsout)
            SVsout{wi} = cell(num_classes,1);
        end
        
        % create temporary variables to store input/labels for all workers
        % in current cascade level
        
        
        % train current level of SVMs in parallel
        parfor labindex = 1:worker_count
            for ci = 1:num_classes
                labels_cls_this = labels_cls(SVsin{labindex}{ci},ci);
                input_this = input(:,SVsin{labindex}{ci});
                input_this = input_this';
                libsvm_cascade = svmtrain(labels_cls_this, input_this, sprintf(' -t 0 -c %f', c));
                % convert support vectors from model (which are full
                % vectors) to index references in input
                SVidxs = ismember(input_this,libsvm_cascade.SVs,'rows');
                SVidxs = find(SVidxs);
                for i = 1:length(SVidxs)
                    SVidxs(i) = SVsin{labindex}{ci}(i);
                end
                SVsout{labindex}{ci} = SVidxs;
            end
        end
        
        % calculate number of workers for next iteration and then continue
        worker_count = worker_count/2;
    end
    
    clear input_this labels_cls_this libsvm_cascade;
    
    % deal with final level of the cascade
    % (the following code is the same as regular libsvm, just with fewer
    % support vectors)
    
    % prepare temporary output model storage variables
    libsvm = cell(1,num_classes);
    libsvm_flipscore = zeros(1,num_classes);
    
    % convert SVsout to a single level cell to make it inputtable to parfor
    SVsin_flat = cell(1,num_classes);
    for ci = 1:num_classes
        SVsin_flat{ci} = [SVsout{ci}{1}; SVsout{ci}{2}];
    end
    clear SVsout;
    
    % train models for each class in turn
    input = input';
    parfor ci = 1:num_classes    
        fprintf('FINAL RUN -- Class: %d, SVs: %d\n',worker_count,wi,ci,length(SVsin_flat{ci}));
        libsvm{ci} = svmtrain(labels_cls(SVsin_flat{ci},ci), input(SVsin_flat{ci},:), ...
            sprintf(' -t 0 -c %f', obj.c)); %#ok<PFBNS>
        % in single class classification, first label encountered is
        % assigned to +1, so if the opposite is true in the label set,
        % set a flag in the libsvm struct to indicate this
        libsvm_flipscore(ci) = (labels_cls(SVsin_flat{ci}(1),ci) == -1);
    end
    
    % copy across trained model
    obj.model = struct;
    obj.model.libsvm = libsvm;
    obj.model.libsvm_flipscore = libsvm_flipscore;
    
    % apply bias multiplier if required
    if obj.bias_mul ~= 1
        for i = 1:length(obj.model.libsvm)
            obj.model.libsvm{i}.rho = ...
                obj.bias_multiplier*obj.model.libsvm{i}.rho;
        end
    end
end

