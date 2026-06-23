% plot_welch_100s.m
%
% Load one converted tracking dataset MAT file and plot a 100-second Welch PSD
% for one component: X, Y, or A.
%
% Usage:
%   1) Edit matFile and component below
%   2) Run the script
%
% Notes:
% - Uses 100 s windows for pwelch
% - Interpolates over NaNs before spectral estimation
% - Removes mean before computing PSD
% - Plots one curve per block

clear; clc; close all;

%% ---------------- User settings ----------------
matFile = '../data/11topo/11topo_converted.mat';   % path to converted MAT file
component = 'X';                    % 'X', 'Y', or 'A'
maxFreqHz = [];                     % [] for full range, or e.g. 1
minValidFraction = 0.5;             % require at least this fraction finite in a trace
detrendMode = 'constant';           % 'constant' or 'linear'
showLegend = true;
%% ------------------------------------------------

S = load(matFile);
if ~isfield(S, 'dataset')
    error('MAT file does not contain variable "dataset".');
end
D = S.dataset;

if ~isfield(D, 'frameTimes_s')
    error('dataset.frameTimes_s is missing.');
end

t = D.frameTimes_s(:);
if numel(t) < 2
    error('Not enough time samples in dataset.frameTimes_s.');
end

dt = median(diff(t), 'omitnan');
Fs = 1 / dt;

switch upper(component)
    case 'X'
        if ~isfield(D, 'X'), error('dataset.X is missing.'); end
        M = D.X;
        yLabelText = 'PSD of X (px^2/Hz)';
        titleText = 'Welch PSD, 100 s window: X';
    case 'Y'
        if ~isfield(D, 'Y'), error('dataset.Y is missing.'); end
        M = D.Y;
        yLabelText = 'PSD of Y (px^2/Hz)';
        titleText = 'Welch PSD, 100 s window: Y';
    case 'A'
        if ~isfield(D, 'A'), error('dataset.A is missing.'); end
        M = D.A;
        yLabelText = 'PSD of angle ((units)^2/Hz)';
        titleText = 'Welch PSD, 100 s window: A';
    otherwise
        error('component must be ''X'', ''Y'', or ''A''.');
end

if size(M,1) ~= numel(t)
    error('Component row count (%d) does not match number of time points (%d).', ...
        size(M,1), numel(t));
end

% Block labels
if isfield(D, 'blockColors')
    labels = string(D.blockColors);
    labels = labels(:);
else
    labels = "block_" + string(1:size(M,2)).';
end
if numel(labels) < size(M,2)
    labels(end+1:size(M,2)) = "block_" + string(numel(labels)+1:size(M,2)).';
end

% Welch settings: 100-second window
winSec = 100;
nwin = max(8, round(winSec * Fs));
noverlap = round(0.5 * nwin);

fprintf('Loaded: %s\n', matFile);
fprintf('Component: %s\n', upper(component));
fprintf('Frames: %d\n', size(M,1));
fprintf('Blocks: %d\n', size(M,2));
fprintf('Median dt: %.6f s\n', dt);
fprintf('Estimated Fs: %.6f Hz\n', Fs);
fprintf('Welch window: %d samples (%.3f s)\n', nwin, nwin/Fs);
fprintf('Overlap: %d samples\n', noverlap);

figure('Color', 'w');
hold on;

nPlotted = 0;
legendEntries = strings(0,1);

for k = 1:size(M,2)
    x = M(:,k);
    valid = isfinite(x);
    fracValid = mean(valid);

    if fracValid < minValidFraction
        fprintf('Skipping block %d (%s): only %.1f%% valid\n', ...
            k, labels(k), 100*fracValid);
        continue;
    end

    % Fill missing samples by interpolation for Welch estimate
    xi = fillmissing(x, 'linear', 'SamplePoints', t);
    xi = fillmissing(xi, 'nearest');

    % Remove trend / mean
    switch lower(detrendMode)
        case 'constant'
            xi = detrend(xi, 'constant');
        case 'linear'
            xi = detrend(xi, 'linear');
        otherwise
            error('Unsupported detrendMode: %s', detrendMode);
    end

    % Welch PSD
    [Pxx, f] = pwelch(xi, nwin, noverlap, [], Fs);

    if ~isempty(maxFreqHz)
        keep = (f <= maxFreqHz);
        fPlot = f(keep);
        PPlot = Pxx(keep);
    else
        fPlot = f;
        PPlot = Pxx;
    end

    plot(fPlot, PPlot, 'LineWidth', 1.2);
    nPlotted = nPlotted + 1;
    legendEntries(end+1,1) = labels(k); %#ok<SAGROW>
end

grid on;
xlabel('Frequency (Hz)');
ylabel(yLabelText);
title(titleText, 'Interpreter', 'none');

if showLegend && nPlotted > 0
    legend(cellstr(legendEntries), 'Location', 'eastoutside', 'Interpreter', 'none');
end

hold off;

%% Optional second figure: amplitude spectrum estimate from PSD
% This is not a raw FFT magnitude; it is sqrt(PSD), which is often easier to read.
figure('Color', 'w');
hold on;

nPlotted2 = 0;
legendEntries2 = strings(0,1);

for k = 1:size(M,2)
    x = M(:,k);
    valid = isfinite(x);
    fracValid = mean(valid);

    if fracValid < minValidFraction
        continue;
    end

    xi = fillmissing(x, 'linear', 'SamplePoints', t);
    xi = fillmissing(xi, 'nearest');

    switch lower(detrendMode)
        case 'constant'
            xi = detrend(xi, 'constant');
        case 'linear'
            xi = detrend(xi, 'linear');
    end

    [Pxx, f] = pwelch(xi, nwin, noverlap, [], Fs);
    Axx = sqrt(Pxx);

    if ~isempty(maxFreqHz)
        keep = (f <= maxFreqHz);
        fPlot = f(keep);
        APlot = Axx(keep);
    else
        fPlot = f;
        APlot = Axx;
    end

    plot(fPlot, APlot, 'LineWidth', 1.2);
    nPlotted2 = nPlotted2 + 1;
    legendEntries2(end+1,1) = labels(k); %#ok<SAGROW>
end

grid on;
xlabel('Frequency (Hz)');
ylabel('sqrt(PSD)');
title(sprintf('Welch amplitude-style spectrum, 100 s window: %s', upper(component)), ...
    'Interpreter', 'none');

if showLegend && nPlotted2 > 0
    legend(cellstr(legendEntries2), 'Location', 'eastoutside', 'Interpreter', 'none');
end

hold off;
