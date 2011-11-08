classdef VQEncoder < handle & featpipem.encoding.GenericEncoder
    %VQENCODER Bag-of-word histogram computation using the VQ method (hard assignment)
    
    properties
        norm_type % 'l1' or 'l2'
        max_comps % -1 for exact
    end
    
    properties(SetAccess=protected)
        codebook_
        kdtree_
    end
    
    methods
        function obj = VQEncoder(codebook)
            % set default parameter values
            obj.norm_type = 'l1';
            obj.max_comps = -1;
            
            % setup encoder
            obj.codebook_ = codebook;
            obj.kdtree_ = vl_kdtreebuild(obj.codebook_);
        end
        function dim = get_input_dim(obj)
            dim = size(obj.codebook_,1);
        end
        function dim = get_output_dim(obj)
            dim = size(obj.codebook_,2);
        end
        code = encode(obj, feats)
    end
    
end

