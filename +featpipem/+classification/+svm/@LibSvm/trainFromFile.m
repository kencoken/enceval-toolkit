function trainFromFile(obj, files, labels, labelNames)
%TRAIN Testing function for LIBSVM (using dual formulation)
%   Refer to GenericSVM for interface definition

    % compute kernel matrix iteratively
    obj.K = zeros(length(labels), length(labels));
    for f1 = 1:length(files)
        fprintf('Computing K for datafile %d of %d\n', f1, ...
                length(files));
        load(files{f1});
        chunk1 = chunk;
        index1 = index;
        for f2 = 1:length(files)
            fprintf('Processing %s vs %s...\n', files{f1}, ...
                    files{f2});
            load(files{f2});
            k = chunk1'*chunk;
            K(index1,index) = k;
            K(index,index1) = k';
        end
    end
    disp('Kernel matrix computed');

end

