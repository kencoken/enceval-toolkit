classdef PCAFeatExtractor < handle & featpipem.features.GenericFeatExtractor
    %PCAFEATEXTRACTOR Class wrapping other feature extractor with PCA dim
    %reduction
    
    properties(SetAccess = private)
        featextr
        out_dim
        proj
    end
    
    methods
        function obj = PCAFeatExtractor(featextr, out_dim)
            obj.featextr = featextr;
            obj.out_dim = out_dim;
            proj = [];
        end
        train_proj(obj, imlist, varargin)
        save_proj(obj, fname)
        load_proj(obj, fname)
        [feats, frames] = compute(obj, im)
    end
    
end

