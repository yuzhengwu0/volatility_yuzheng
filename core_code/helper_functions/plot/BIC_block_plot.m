%% plot BIC-selected beta blocks
%
% Layout:
%
% coh == 0:
%       left  = all
%       right = blank
%
% other coherence levels:
%       left  = incorrect
%       right = correct

clear; clc; close all;

load('../figure/RES_BIC_beta_mixed_accuracy_layout.mat');

%% settings

termLabels = ["rt", "cond", "Vt", "Vt x cond"];

rowColors = [
    0.45 0.45 0.45;   % rt gray
    0.90 0.72 0.12;   % cond yellow
    0.75 0.10 0.10;   % Vt dark red
    0.10 0.30 0.80    % Vt x cond dark blue
];

minAlpha = 0.02;
maxAlpha = 1.00;

showSignText = true;
showBetaText = false;

nRows = numel(coh_levels);
nCols = 2;

globalMin = globalBetaMin;
globalMax = globalBetaMax;

if isempty(globalMin) || isempty(globalMax)

    globalMin = 0;
    globalMax = 1;

end

if globalMax == globalMin
    globalMax = globalMin + eps;
end

%% plot tiledlayout

figWidth = 1500;
figHeight = max(500, 300 * nRows);

fig = figure( ...
    'Color', 'w', ...
    'Position', [80 50 figWidth figHeight]);

tl = tiledlayout( ...
    nRows, ...
    nCols, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

sgtitle(tl, sprintf( ...
    ['BIC-selected beta blocks | ', ...
     'opacity = global |beta| | ', ...
     'min = %.3f, max = %.3f'], ...
    globalMin, globalMax), ...
    'FontWeight', 'bold', ...
    'FontSize', 16);

for ci = 1:nRows

    for ai = 1:nCols

        ax = nexttile(tl);
        hold(ax, 'on');

        %% ----- blank coh == 0 right panel -----

        if isfield(RES(ci, ai), 'isBlank') && ...
                RES(ci, ai).isBlank

            axis(ax, 'off');

            continue;

        end

        %% ----- retrieve selected betas -----

        selectedBetas = RES(ci, ai).selectedBetas;
        includedTerm = RES(ci, ai).includedTerm;

        nTerms = size(selectedBetas, 1);
        nBins = size(selectedBetas, 2);

        %% ----- empty grid -----

        for t = 1:nBins

            for k = 1:nTerms

                rectangle(ax, ...
                    'Position', [t-0.5, k-0.5, 1, 1], ...
                    'FaceColor', [1 1 1], ...
                    'EdgeColor', [0.86 0.86 0.86], ...
                    'LineWidth', 0.35);

            end
        end

        %% ----- colored blocks -----

        for t = 1:nBins

            for k = 1:nTerms

                if includedTerm(k, t) && ...
                        ~isnan(selectedBetas(k, t))

                    betaVal = selectedBetas(k, t);
                    absVal = abs(betaVal);

                    rawAlpha = ...
                        (absVal - globalMin) / ...
                        (globalMax - globalMin);

                    rawAlpha = ...
                        max(0, min(rawAlpha, 1));

                    alphaVal = ...
                        minAlpha + ...
                        (maxAlpha - minAlpha) * rawAlpha;

                    rectangle(ax, ...
                        'Position', [t-0.5, k-0.5, 1, 1], ...
                        'FaceColor', rowColors(k, :), ...
                        'FaceAlpha', alphaVal, ...
                        'EdgeColor', [0.86 0.86 0.86], ...
                        'LineWidth', 0.35);

                    %% text label

                    if showBetaText

                        labelTxt = ...
                            sprintf('%.2f', betaVal);

                    elseif showSignText

                        if betaVal > 0

                            labelTxt = '+';

                        elseif betaVal < 0

                            labelTxt = '−';

                        else

                            labelTxt = '0';

                        end

                    else

                        labelTxt = '';

                    end

                    if ~isempty(labelTxt)

                        if alphaVal > 0.65

                            txtColor = [1 1 1];

                        else

                            txtColor = [0 0 0];

                        end

                        text(ax, ...
                            t, ...
                            k, ...
                            labelTxt, ...
                            'HorizontalAlignment', 'center', ...
                            'VerticalAlignment', 'middle', ...
                            'FontSize', 7, ...
                            'FontWeight', 'bold', ...
                            'Color', txtColor);

                    end
                end
            end
        end

        %% ----- axis -----

        xTicks = unique([1, 5:5:nBins, nBins]);

        set(ax, ...
            'YDir', 'normal', ...
            'XTick', xTicks, ...
            'YTick', 1:nTerms, ...
            'YTickLabel', termLabels, ...
            'FontSize', 9, ...
            'TickLength', [0 0]);

        xlim(ax, [0.5, nBins + 0.5]);
        ylim(ax, [0.5, nTerms + 0.5]);

        box(ax, 'on');

        %% ----- title based on condition -----

        switch lower(RES(ci, ai).acc)

            case 'all'

                thisTitle = 'All';

            case 'incorr'

                thisTitle = 'Incorrect';

            case 'corr'

                thisTitle = 'Correct';

            otherwise

                thisTitle = RES(ci, ai).acc;

        end

        title(ax, ...
            sprintf('%s | n = %d', ...
                thisTitle, ...
                RES(ci, ai).nTrials), ...
            'FontWeight', 'bold');

        %% ----- row labels -----

        %% ----- row labels -----
        
        if ai == 1
        
            ylabel(ax, ...
                sprintf('coh = %d', RES(ci, ai).coh), ...
                'FontWeight', 'bold');
        
        else
        
            ylabel(ax, '');
            ax.YTickLabel = [];
        
        end

        %% ----- bottom x labels only -----

        if ci == nRows

            xlabel(ax, 'Time bin');

        else

            xlabel(ax, '');
            ax.XTickLabel = [];

        end

    end
end

%% save figure

outDir = '../figure';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

outName = ...
    'BIC_selected_beta_blocks_coh0_all_other_corr_incorr.png';

outPath = fullfile(outDir, outName);

exportgraphics( ...
    fig, ...
    outPath, ...
    'Resolution', 300);

fprintf('Saved tiled figure to:\n%s\n', outPath);