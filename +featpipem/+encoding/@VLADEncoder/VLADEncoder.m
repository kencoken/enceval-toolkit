classdef VLADEncoder < handle & featpipem.encoding.GenericEncoder
    %VQENCODER Bag-of-word histogram computation using the VQ method (hard assignment)
    
    properties
        subencoder
    end
    
    methods
        function obj = VLADEncoder(subencoder)
            
            obj.subencoder = subencoder;
            
           
        end
        
        function dim = get_input_dim(obj)
            
            dim = obj.subencoder.get_input_dim();
            
        end
        
        function dim = get_output_dim(obj)
            
            dim = obj.subencoder.get_output_dim() * obj.get_input_dim();
            
        end
        
        code = encode(obj, feats)
    end
    
end

