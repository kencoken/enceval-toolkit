function [map, file] = findbestperf(pathloc)
%FINDBESTPERF Summary of this function goes here
%   Detailed explanation goes here

    if nargin > 0
        files = getFilenames(pathloc, {'mat'});
    else
        [files, path] = uigetfile('*.mat','Select result files',pwd,'MultiSelect','on');
        for i = 1:length(files)
            files{i} = fullfile(path, files{i});
        end
    end
    
    maxMap = 0;
    maxCFile = '';
    
    for i = 1:length(files)
        file = load(files{i});
        fmap = mean(file.results.res{1});
        if fmap > maxMap
            maxMap = fmap;
            maxCFile = files{i};
        end
    end
    
    map = maxMap;
    file = maxCFile;
end

