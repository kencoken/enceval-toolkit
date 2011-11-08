function [est_label, scoremat] = test(obj, input)
%TEST Training function for LIBLINEAR
%   Refer to GenericSVM for interface definition

    % ensure a model has been trained
    if isempty(obj.model)
        error('A SVM model has yet to be trained');
    end
    
    % test models for each class in turn
    scoremat = obj.model.w'*input + obj.model.b'*ones(1,size(input,2));
    
    [est_label est_label] = max(scoremat, [], 1); %#ok<ASGLU>
end

