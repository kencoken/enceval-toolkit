function codebook = loadcodebook(codebkgen, prms)
%LOADCODEBOOK Summary of this function goes here
%   Detailed explanation goes here

if exist(prms.codebook,'file')
    load(prms.codebook);
else
    imfiles = cell(length(prms.imdb.images.name),1);
    for i = 1:length(prms.imdb.images.name)
        imfiles{i} = fullfile(prms.paths.dataset, prms.imdb.images.name{i});
    end
    
    % do training...
    codebook = codebkgen.train(imfiles);
    
    save(prms.codebook,'codebook','codebkgen');
end

end

