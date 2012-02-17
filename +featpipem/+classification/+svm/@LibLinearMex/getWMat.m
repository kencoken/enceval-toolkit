function WMat = getWMat(obj)
%GETWMAT Summary of this function goes here
%   Detailed explanation goes here

    % ensure a model has been trained
    if isempty(obj.model)
        error('A SVM model has yet to be trained');
    end
    
    WMat = obj.model;
end

