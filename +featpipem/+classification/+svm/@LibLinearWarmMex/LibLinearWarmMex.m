classdef LibLinearWarmMex < handle & featpipem.classification.svm.LinearSvm
    %LIBLINEARWARMMEX Train an SVM classifier using the LIBLINEAR library
    %(training code is in C)
    
    properties
        % svm parameters
        c            % SVM C parameter
        c_handle     % handle to wrapped c++ class
    end
    
    methods
        function obj = LibLinearWarmMex(varargin)
            opts.c = 10;
            [opts, varargin] =  vl_argparse(opts, varargin);
            obj.c = opts.c;
            
            obj = manage_class(obj, 'init');
            
            % load in the model if provided
%             modelstore.model = [];
%             vl_argparse(modelstore, varargin);
%             obj.model = modelstore.model;
        end
        function delete(obj)
            manage_class(obj,'clear');
        end
        function train(obj, input, labels, init_w)
            input = single(input);
            init_w = single(init_w);
            obj = mextrain(obj, input, labels, init_w);
        end
        function [est_label, scoremat] = test(obj, input)
            [est_label, scoremat] = mextest(obj, input);
            est_label = est_label';
            scoremat = scoremat';
        end
        WMat = getWMat(obj)
        
    end
    
end

