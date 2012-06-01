classdef GenericFeatExtractor < handle
    %GENERICFEATEXTRACTOR Generic interface for extracting image features
    
    properties
    end
    
    properties(SetAccess = private)
        outdim
    end
    
    methods(Abstract)
        [feats, frames] = compute(obj, im)
    end
    
end

