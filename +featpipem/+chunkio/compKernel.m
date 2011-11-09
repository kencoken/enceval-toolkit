function K = compKernel(chunk_files)
%COMPKERNEL Compute kernel matrix from feature chunk files
%   chunk_files -  cell array of paths to chunk files to use to compute K,
%                  with assumed complete monotonically increasing set of
%                  indeces

% process and check input arguments -----

if ~iscell(chunk_files)
    error(['chunk_files must be either a cell of strings, or cell ' ...
        'of cell of strings (to support multiple input sets)']);
end
% if the input is just a cell array of strings, nest it in a second
% level (as this is the form used when accepting multiple input test
% sets)
if ischar(chunk_files{1})
    chunk_files = {chunk_files};
end

% preallocate kernel matrix
ch = load(chunk_files{1}{1});
size_est = 0;
for i = 1:length(chunk_files)
    size_est = size_est + size(ch.chunk, 2)*length(chunk_files{i});
end
K = zeros(size_est);
clear ch size_est;

idxoffseti = 0;
maxidxi = 0;
% keep track of largest index stored in kernel matrix (used for
% preallocation)
maxidx_ker = 0;

% iterate over first chunkfile
for si = 1:length(chunk_files)
    idxoffseti = idxoffseti+maxidxi;
    maxidxi = 0;
    for ci = 1:length(chunk_files{si})
        fprintf('Computing datafile for chunk %d of %d (in set %d of %d)\n', ...
            ci, length(chunk_files{si}), si, length(chunk_files));
        
        ch1 = load(chunk_files{si}{ci});
        % apply index offset if required
        ch1.index = ch1.index + idxoffseti;
        % store maxidxi for current set (to calculate offset for next set)
        if ch1.index(end) > maxidxi, maxidxi = ch1.index(end); end
        % store absolute max index to aid resizing of kernel matrix at end
        if ch1.index(end) > maxidx_ker, maxidx_ker = ch1.index(end); end
        % iterate over second chunkfile
        idxoffsetj = 0;
        maxidxj = 0;
        for sj = 1:length(chunk_files)
            idxoffsetj = idxoffsetj+maxidxj;
            maxidxj = 0;
            for cj = 1:length(chunk_files{sj})
                fprintf('  Processing %s vs. %s\n', chunk_files{si}{ci}, chunk_files{sj}{cj});
                ch2 = load(chunk_files{sj}{cj});
                % apply index offset if required
                ch2.index = ch2.index + idxoffsetj;
                % store maxidxj for current set (to calculate offset for next set)
                if ch2.index(end) > maxidxj, maxidxj = ch2.index(end); end
                % TEMPORARY CODE - ensure all codes are L2 normalized
                for i = 1:size(ch1.chunk,2)
                    ch1.chunk(:,i) = ch1.chunk(:,i)/norm(ch1.chunk(:,i),2);
                end
                for i = 1:size(ch2.chunk,2)
                    ch2.chunk(:,i) = ch2.chunk(:,i)/norm(ch2.chunk(:,i),2);
                end
                % do computation of sub-part of kernel matrix
                k = ch1.chunk'*ch2.chunk;
                K(ch1.index, ch2.index) = k;
                K(ch2.index, ch1.index) = k';
            end
        end
    end
end

% use maxidx_ker to resize output matrix to be correct size
K = K(1:maxidx_ker,1:maxidx_ker);

fprintf('Kernel matrix computed\n');


end

