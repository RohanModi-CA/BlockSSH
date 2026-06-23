% plot_all_blocks_timeseries.m
clear; clc; close all;

load ../data/10MassNew/10MassNew_legacy.mat

% Make sure time is a column
t = t(:);

% Basic checks
if size(xR,1) ~= numel(t)
    error('xR row count (%d) does not match length(t) (%d).', size(xR,1), numel(t));
end
if size(yR,1) ~= numel(t)
    error('yR row count (%d) does not match length(t) (%d).', size(yR,1), numel(t));
end
if exist('oR','var') && ~isempty(oR) && size(oR,1) ~= numel(t)
    error('oR row count (%d) does not match length(t) (%d).', size(oR,1), numel(t));
end

nBlocks = size(xR,2);

% Labels
if exist('blockColors','var') && numel(blockColors) >= nBlocks
    labels = string(blockColors(:));
else
    labels = "block_" + string(1:nBlocks).';
end

% ---- X positions ----
figure('Color','w','Name','X time series');
plot(t, xR, 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('x position');
title('All block x-position time series');
legend(cellstr(labels), 'Location','eastoutside', 'Interpreter','none');

% ---- Y positions ----
figure('Color','w','Name','Y time series');
plot(t, yR, 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('y position');
title('All block y-position time series');
legend(cellstr(labels), 'Location','eastoutside', 'Interpreter','none');

% ---- Orientation, if present ----
if exist('oR','var') && ~isempty(oR)
    figure('Color','w','Name','Angle time series');
    plot(t, oR, 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel('angle');
    title('All block angle time series');
    legend(cellstr(labels), 'Location','eastoutside', 'Interpreter','none');
end
