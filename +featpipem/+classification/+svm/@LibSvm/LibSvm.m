classdef LibSvm < handle & featpipem.classification.svm.GenericSvm
    %LIBSVM Train an SVM classifier using the LIBSVM library
    
    properties
        % svm parameters
        c            % SVM C parameter
        bias_mul     % SVM bias multiplier
    end
    
    methods
        function obj = LibSvm(varargin)
            opts.c = 10;
            opts.bias_mul = 1;
            [opts, varargin] =  vl_argparse(opts, varargin);
            vl_override(obj, opts);
            
            % load in the model if provided
            modelstore.model = [];
            vl_argparse(modelstore, varargin);
            obj.model = modelstore.model;
        end
        train(obj, input, labels)
        [est_label, scoremat] = test(obj, input)
        
    end
    
end
