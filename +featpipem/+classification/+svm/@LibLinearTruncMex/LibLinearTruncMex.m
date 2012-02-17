classdef LibLinearTruncMex < handle & featpipem.classification.svm.LinearSvm
    %LIBLINEARTRUNCMEX Train an SVM classifier using the LIBLINEAR library
    %(training code is in C)
    
    properties
        % svm parameters
        c            % SVM C parameter
        c_handle     % handle to wrapped c++ class
        sparsity     % proportion of zeros in classifier w
    end
    
    methods
        function obj = LibLinearTruncMex(varargin)
            opts.c = 10;
            opts.sparsity = 0.0;
            [opts, varargin] =  vl_argparse(opts, varargin);
            obj.c = opts.c;
            obj.sparsity = opts.sparsity;
            
            obj = manage_class(obj, 'init');
            
            % load in the model if provided
%             modelstore.model = [];
%             vl_argparse(modelstore, varargin);
%             obj.model = modelstore.model;
        end
        function delete(obj)
            manage_class(obj,'clear');
        end
        function train(obj, input, labels)
            input = single(input);
            obj = mextrain(obj, input, labels);
        end
        function [est_label, scoremat] = test(obj, input)
            [est_label, scoremat] = mextest(obj, input);
            est_label = est_label';
            scoremat = scoremat';
        end
        WMat = getWMat(obj)
        
    end
    
end

