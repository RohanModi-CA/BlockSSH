function S = plot_tracking_dataset(datasetDir)
%PLOT_TRACKING_DATASET Read one tracking dataset folder and plot x/y/a timeseries.
%
% Usage:
%   S = plot_tracking_dataset('/path/to/track/data/11topo');
%
% Requires:
%   parsemsgpack.m from matlab-msgpack on the MATLAB path
%
% Expected files inside datasetDir:
%   params_bottom.json                           (optional)
%   track1.msgpack                               (optional)
%   components/x/track2_permanence.msgpack       (recommended)
%   components/y/track2_permanence.msgpack       (recommended)
%   components/a/track2_permanence.msgpack       (recommended)
%
% Returns:
%   S : struct with loaded data, summaries, and file paths

    arguments
        datasetDir (1,1) string
    end

    datasetDir = string(datasetDir);
    if ~isfolder(datasetDir)
        error('Dataset folder not found: %s', datasetDir);
    end

    % ---------- Paths ----------
    paths.datasetDir = datasetDir;
    paths.params     = fullfile(datasetDir, 'params_bottom.json');
    paths.track1     = fullfile(datasetDir, 'track1.msgpack');
    paths.x          = fullfile(datasetDir, 'components', 'x', 'track2_permanence.msgpack');
    paths.y          = fullfile(datasetDir, 'components', 'y', 'track2_permanence.msgpack');
    paths.a          = fullfile(datasetDir, 'components', 'a', 'track2_permanence.msgpack');

    fprintf('\n=== Tracking dataset summary ===\n');
    fprintf('Dataset folder: %s\n', datasetDir);

    % ---------- Optional params ----------
    params = struct();
    if isfile(paths.params)
        params = jsondecode(fileread(paths.params));
        fprintf('Found params_bottom.json\n');
    else
        fprintf('params_bottom.json: not found\n');
    end

    % ---------- Load verified permanence outputs ----------
    tx = [];
    ty = [];
    ta = [];

    if isfile(paths.x), tx = load_msgpack_file(paths.x); end
    if isfile(paths.y), ty = load_msgpack_file(paths.y); end
    if isfile(paths.a), ta = load_msgpack_file(paths.a); end

    if isempty(tx) && isempty(ty) && isempty(ta)
        error('No track2_permanence.msgpack files found under components/x|y|a');
    end

    % Use x as canonical metadata when available, otherwise fall back.
    tmain = first_nonempty(tx, ty, ta);

    X = get_numeric_field(tx, 'xPositions', []);
    Y = get_numeric_field(ty, 'xPositions', []);
    A = get_numeric_field(ta, 'xPositions', []);

    t = get_numeric_field(tmain, 'frameTimes_s', []);
    frameNums = get_numeric_field(tmain, 'frameNumbers', []);
    blockColors = get_text_list(tmain, 'blockColors');

    % Normalize shapes to [nFrames x nBlocks]
    X = ensure_2d_numeric(X);
    Y = ensure_2d_numeric(Y);
    A = ensure_2d_numeric(A);
    t = make_column(t);
    frameNums = make_column(frameNums);

    nFramesCandidates = [size(X,1), size(Y,1), size(A,1), numel(t), numel(frameNums)];
    nFramesCandidates = nFramesCandidates(nFramesCandidates > 0);
    nFrames = mode(nFramesCandidates);

    if isempty(t)
        t = (0:nFrames-1).';
    end
    if isempty(frameNums)
        frameNums = (0:nFrames-1).';
    end

    if ~isempty(X) && size(X,1) ~= nFrames
        warning('X has %d rows, expected %d', size(X,1), nFrames);
    end
    if ~isempty(Y) && size(Y,1) ~= nFrames
        warning('Y has %d rows, expected %d', size(Y,1), nFrames);
    end
    if ~isempty(A) && size(A,1) ~= nFrames
        warning('A has %d rows, expected %d', size(A,1), nFrames);
    end

    nBlocks = max([size(X,2), size(Y,2), size(A,2), numel(blockColors)]);
    if isempty(blockColors)
        blockColors = compose("block_%02d", 1:nBlocks);
    else
        blockColors = string(blockColors(:)');
        if numel(blockColors) < nBlocks
            blockColors(end+1:nBlocks) = compose("block_%02d", numel(blockColors)+1:nBlocks);
        end
    end

    % ---------- Optional raw track1 summary ----------
    track1 = [];
    track1Summary = struct();
    if isfile(paths.track1)
        track1 = load_msgpack_file(paths.track1);
        track1Summary = summarize_track1(track1);
        fprintf('Found track1.msgpack\n');
    else
        fprintf('track1.msgpack: not found\n');
    end

    % ---------- Print high-level info ----------
    fprintf('\n--- Verified track2 summary ---\n');
    fprintf('Frames: %d\n', nFrames);
    fprintf('Blocks/columns: %d\n', nBlocks);
    fprintf('Time span: %.6g to %.6g s', t(1), t(end));
    if numel(t) >= 2
        dt = diff(t);
        dt = dt(isfinite(dt));
        if ~isempty(dt)
            fprintf('  (median dt = %.6g s, ~%.6g Hz)', median(dt), 1/median(dt));
        end
    end
    fprintf('\n');
    fprintf('Frame numbers: %g to %g\n', frameNums(1), frameNums(end));

    if ~isempty(blockColors)
        fprintf('Block labels/colors: %s\n', strjoin(cellstr(blockColors), ', '));
    end

    print_component_summary('x', X, blockColors);
    print_component_summary('y', Y, blockColors);
    print_component_summary('a', A, blockColors);

    if ~isempty(fieldnames(track1Summary))
        fprintf('\n--- Raw track1 summary ---\n');
        fprintf('Frames in track1: %d\n', track1Summary.nFrames);
        fprintf('Frame number range: %g to %g\n', track1Summary.firstFrameNumber, track1Summary.lastFrameNumber);
        fprintf('Detections/frame: min=%g  median=%g  max=%g\n', ...
            track1Summary.minDetectionsPerFrame, ...
            track1Summary.medianDetectionsPerFrame, ...
            track1Summary.maxDetectionsPerFrame);
        if isfield(track1Summary, 'colorsSeen') && ~isempty(track1Summary.colorsSeen)
            fprintf('Colors seen in track1: %s\n', strjoin(cellstr(track1Summary.colorsSeen), ', '));
        end
    end

    if ~isempty(fieldnames(params))
        fprintf('\n--- Params summary ---\n');
        print_if_present(params, 'crop_top');
        print_if_present(params, 'crop_bottom');
        print_if_present(params, 'time_start_s');
        print_if_present(params, 'time_end_s');
        print_if_present(params, 'min_area');
        print_if_present(params, 'max_area');
        print_if_present(params, 'minWhiteCoverageFraction');
        print_if_present(params, 'ringOuterRadius');
        print_if_present(params, 'ccConnectivity');
    end

    % ---------- Plot ----------
    fig = figure('Name', sprintf('Tracking dataset: %s', datasetDir), 'Color', 'w');
    tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_component(nexttile(tl), t, X, blockColors, 'x position (px)', 'Component x');
    plot_component(nexttile(tl), t, Y, blockColors, 'y position (px)', 'Component y');
    plot_component(nexttile(tl), t, A, blockColors, 'angle (rad)',     'Component a');

    xlabel(tl, 'Time (s)');

    % ---------- Return ----------
    S = struct();
    S.paths = paths;
    S.params = params;
    S.track1 = track1;
    S.track1Summary = track1Summary;
    S.track2_x = tx;
    S.track2_y = ty;
    S.track2_a = ta;
    S.t = t;
    S.frameNumbers = frameNums;
    S.blockColors = blockColors;
    S.X = X;
    S.Y = Y;
    S.A = A;
    S.nFrames = nFrames;
    S.nBlocks = nBlocks;
end


function obj = load_msgpack_file(path)
    bytes = fileread_uint8(path);
    obj = parsemsgpack(bytes);
    obj = msgpack_to_matlab(obj);
end

function bytes = fileread_uint8(path)
    fid = fopen(path, 'rb');
    if fid < 0
        error('Could not open file: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    bytes = fread(fid, Inf, '*uint8');
end

function out = msgpack_to_matlab(in)
    % Recursively convert matlab-msgpack outputs into MATLAB-native values.
    if isa(in, 'containers.Map')
        keys = in.keys;
        out = struct();
        for i = 1:numel(keys)
            k = keys{i};
            out.(matlab.lang.makeValidName(char(k))) = msgpack_to_matlab(in(k));
        end
    elseif iscell(in)
        out = cell(size(in));
        for i = 1:numel(in)
            out{i} = msgpack_to_matlab(in{i});
        end

        % If every cell is a scalar struct with same fields, convert to struct array.
        if ~isempty(out) && all(cellfun(@(c) isstruct(c) && isscalar(c), out))
            f0 = sort(fieldnames(out{1}));
            same = all(cellfun(@(c) isequal(sort(fieldnames(c)), f0), out));
            if same
                out = [out{:}];
            end
        end
    else
        out = in;
    end
end

function v = get_numeric_field(s, fieldName, default)
    v = default;
    if isempty(s) || ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    raw = s.(fieldName);

    if isnumeric(raw) || islogical(raw)
        v = double(raw);
        return;
    end

    if iscell(raw)
        try
            v = cell_to_numeric(raw);
        catch
            v = default;
        end
    end
end

function arr = cell_to_numeric(c)
    if isempty(c)
        arr = [];
        return;
    end

    % vector cell -> numeric vector
    if isvector(c) && all(cellfun(@(x) isnumeric(x) && isscalar(x), c))
        arr = cellfun(@double, c(:));
        return;
    end

    % matrix-like cell -> numeric matrix
    if all(cellfun(@iscell, c))
        nRows = numel(c);
        nCols = numel(c{1});
        arr = nan(nRows, nCols);
        for i = 1:nRows
            row = c{i};
            if ~iscell(row) || numel(row) ~= nCols
                error('Jagged cell matrix');
            end
            for j = 1:nCols
                if isempty(row{j})
                    arr(i,j) = NaN;
                elseif isnumeric(row{j}) && isscalar(row{j})
                    arr(i,j) = double(row{j});
                else
                    error('Non-numeric element inside numeric cell matrix');
                end
            end
        end
        return;
    end

    error('Unsupported cell content for numeric conversion');
end

function txt = get_text_list(s, fieldName)
    txt = strings(1,0);
    if isempty(s) || ~isstruct(s) || ~isfield(s, fieldName)
        return;
    end
    raw = s.(fieldName);
    if isstring(raw)
        txt = raw;
    elseif ischar(raw)
        txt = string(raw);
    elseif iscell(raw)
        txt = strings(1, numel(raw));
        for i = 1:numel(raw)
            txt(i) = string(raw{i});
        end
    end
end

function x = ensure_2d_numeric(x)
    if isempty(x)
        return;
    end
    if isvector(x)
        x = x(:);
    end
    x = double(x);
end

function x = make_column(x)
    if isempty(x)
        x = [];
    else
        x = double(x(:));
    end
end

function out = first_nonempty(varargin)
    out = [];
    for i = 1:nargin
        if ~isempty(varargin{i})
            out = varargin{i};
            return;
        end
    end
end

function print_component_summary(name, M, labels)
    if isempty(M)
        fprintf('%s: not found\n', name);
        return;
    end

    support = sum(isfinite(M), 1);
    frac = support / size(M,1);

    fprintf('\nComponent %s:\n', name);
    fprintf('  Size: %d frames x %d blocks\n', size(M,1), size(M,2));
    fprintf('  Finite entries: %d / %d (%.2f%%)\n', ...
        nnz(isfinite(M)), numel(M), 100 * nnz(isfinite(M)) / max(1, numel(M)));

    for j = 1:size(M,2)
        col = M(:,j);
        good = isfinite(col);
        if any(good)
            vals = col(good);
            fprintf('  %-10s support=%4d (%5.1f%%)  min=%10.4f  max=%10.4f  mean=%10.4f\n', ...
                labels(min(j, numel(labels))), support(j), 100*frac(j), ...
                min(vals), max(vals), mean(vals));
        else
            fprintf('  %-10s support=%4d (%5.1f%%)  all NaN\n', ...
                labels(min(j, numel(labels))), support(j), 100*frac(j));
        end
    end
end

function plot_component(ax, t, M, labels, ylab, ttl)
    axes(ax); %#ok<LAXES>
    cla(ax);

    if isempty(M)
        text(ax, 0.5, 0.5, 'Not available', 'HorizontalAlignment', 'center');
        title(ax, ttl);
        xlabel(ax, 'Time (s)');
        ylabel(ax, ylab);
        grid(ax, 'on');
        return;
    end

    plot(ax, t, M, 'LineWidth', 1.0);
    grid(ax, 'on');
    title(ax, ttl);
    ylabel(ax, ylab);

    if size(M,2) <= 20
        legend(ax, cellstr(labels(1:size(M,2))), 'Location', 'eastoutside', 'Interpreter', 'none');
    end
end

function summary = summarize_track1(track1)
    summary = struct();

    if isempty(track1) || ~isstruct(track1) || ~isfield(track1, 'frames')
        return;
    end

    frames = track1.frames;
    if iscell(frames)
        frames = [frames{:}];
    end
    if isempty(frames)
        return;
    end

    nFrames = numel(frames);
    detCounts = nan(nFrames,1);
    frameNums = nan(nFrames,1);
    colorsSeen = strings(1,0);

    for i = 1:nFrames
        f = frames(i);

        if isfield(f, 'frame_number')
            frameNums(i) = double(f.frame_number);
        end

        if isfield(f, 'detections')
            dets = f.detections;
            if iscell(dets)
                if isempty(dets)
                    dets = struct([]);
                elseif all(cellfun(@isstruct, dets))
                    dets = [dets{:}];
                end
            end

            if isstruct(dets)
                detCounts(i) = numel(dets);
                if ~isempty(dets) && isfield(dets, 'color')
                    c = string({dets.color});
                    colorsSeen = union(colorsSeen, unique(c));
                end
            elseif isempty(dets)
                detCounts(i) = 0;
            end
        end
    end

    summary.nFrames = nFrames;
    summary.firstFrameNumber = min(frameNums(isfinite(frameNums)));
    summary.lastFrameNumber  = max(frameNums(isfinite(frameNums)));
    summary.minDetectionsPerFrame = min(detCounts(isfinite(detCounts)));
    summary.medianDetectionsPerFrame = median(detCounts(isfinite(detCounts)));
    summary.maxDetectionsPerFrame = max(detCounts(isfinite(detCounts)));
    summary.colorsSeen = colorsSeen;
end

function print_if_present(s, fieldName)
    if isfield(s, fieldName)
        val = s.(fieldName);
        if isnumeric(val) && isscalar(val)
            fprintf('%s: %.6g\n', fieldName, val);
        elseif islogical(val) && isscalar(val)
            fprintf('%s: %d\n', fieldName, val);
        elseif ischar(val) || (isstring(val) && isscalar(val))
            fprintf('%s: %s\n', fieldName, string(val));
        end
    end
end
