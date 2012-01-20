classdef LibSvmCascade < handle & featpipem.classification.svm.GenericSvm
    %LIBSVMCASCADE Train an SVM classifier using the LIBSVM library
    
    properties
        % svm parameters
        c            % SVM C parameter
        bias_mul     % SVM bias multiplier
        % cascade parameters
        num_workers
    end
    
    methods
        function obj = LibSvmCascade(varargin)
            opts.c = 10;
            opts.bias_mul = 1;
            opts.num_workers = 8;
            [opts, varargin] =  vl_argparse(opts, varargin);
            obj.c = opts.c;
            obj.bias_mul = opts.bias_mul;
            obj.num_workers = opts.num_workers;
            
            % load in the model if provided
            modelstore.model = [];
            vl_argparse(modelstore, varargin);
            obj.model = modelstore.model;
        end
        train(obj, input, labels)
        [est_label, scoremat] = test(obj, input)
        
    end
    
end

