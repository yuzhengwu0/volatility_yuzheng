function [quarterTbl, figQ, betaBins_q] = plot_quarter_bar(cfg)
% unpack cfg
QUARTER_MODEL_MODE = cfg.QUARTER_MODEL_MODE;
rankIdx            = cfg.rankIdx;
QUARTER_MODEL_NAME = cfg.QUARTER_MODEL_NAME;
modelNames         = cfg.modelNames;
Fitted_models      = cfg.Fitted_models;
QUARTER_TERM_NAME  = cfg.QUARTER_TERM_NAME;

% ===================== choose model =====================
switch QUARTER_MODEL_MODE
    case 'top1'
        qModelIdx = rankIdx(1);

    case 'manual'
        qModelIdx = find(strcmp(modelNames, QUARTER_MODEL_NAME), 1, 'first');
        if isempty(qModelIdx)
            error('QUARTER_MODEL_NAME not found: %s', QUARTER_MODEL_NAME);
        end

    otherwise
        error('Unknown QUARTER_MODEL_MODE: %s', QUARTER_MODEL_MODE);
end

qModelName = modelNames{qModelIdx};
fprintf('\n=== Quarter-bar summary using model: %s ===\n', qModelName);

% ===================== define 4 quarter bins =====================
[K, ~] = size(Fitted_models);

edges = round(linspace(0, K, 5));
qEdges = cell(1,4);
qLabels = {'Q1','Q2','Q3','Q4'};

for q = 1:4
    qEdges{q} = (edges(q)+1):edges(q+1);
end

% ===================== allocate =====================
beta_q     = nan(1,4);
se_q       = nan(1,4);
p_q        = nan(1,4);
n_q        = nan(1,4);
betaBins_q = cell(1,4);

% ===================== summarize existing bin betas =====================
for q = 1:4
    bins_here = qEdges{q};
    beta_bins = nan(1, numel(bins_here));

    for ib = 1:numel(bins_here)
        k = bins_here(ib);

        if ~isfield(Fitted_models(k, qModelIdx), 'g')
            continue;
        end

        g = Fitted_models(k, qModelIdx).g;

        if isempty(g)
            continue;
        end

        coefNames = string(g.CoefficientNames);
        hit = find(coefNames == string(QUARTER_TERM_NAME), 1, 'first');

        if isempty(hit)
            error('Term "%s" is not in model %s.', QUARTER_TERM_NAME, qModelName);
        end

        beta_bins(ib) = g.Coefficients.Estimate(hit);
    end

    beta_bins = beta_bins(~isnan(beta_bins));
    betaBins_q{q} = beta_bins;
    n_q(q) = numel(beta_bins);

    if n_q(q) == 0
        fprintf('Quarter %s skipped: no valid bins.\n', qLabels{q});
        continue;
    elseif n_q(q) == 1
        beta_q(q) = beta_bins(1);
        se_q(q)   = NaN;
        p_q(q)    = NaN;
    else
        beta_q(q) = mean(beta_bins, 'omitnan');
        se_q(q)   = std(beta_bins, 0, 'omitnan') / sqrt(n_q(q));
        [~, p_q(q)] = ttest(beta_bins, 0);
    end
end

% ===================== output path =====================
outPDF_q = fullfile('..', 'figure', ...
    sprintf('QuarterBar_summary_%s_%s.pdf', qModelName, QUARTER_TERM_NAME));

% ===================== plot =====================
figQ = figure('Color','w','Position',[200 200 720 480]);
hold on;

bar(1:4, beta_q, 0.65, 'FaceColor', [0.65 0.65 0.65], 'EdgeColor', 'none');
errorbar(1:4, beta_q, se_q, 'k.', 'LineWidth', 1.2, 'CapSize', 12);

yline(0, 'k--', 'LineWidth', 1);

set(gca, 'XTick', 1:4, 'XTickLabel', qLabels, 'FontSize', 12, 'LineWidth', 1);
xlabel('Within-trial quarter');
ylabel(sprintf('Mean beta: %s', QUARTER_TERM_NAME), 'Interpreter', 'none');
title(sprintf('%s | summary of existing bin betas', qModelName), 'Interpreter', 'none');

% significance stars
yTop = max(beta_q + se_q, [], 'omitnan');
yBot = min(beta_q - se_q, [], 'omitnan');
yrng = yTop - yBot;

if ~isfinite(yrng) || yrng <= 0
    yrng = 1;
end

for q = 1:4
    if ~isnan(p_q(q)) && p_q(q) < 0.01
        text(q, beta_q(q) + se_q(q) + 0.06*yrng, '**', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 18, ...
            'FontWeight', 'bold');

    elseif ~isnan(p_q(q)) && p_q(q) < 0.05
        text(q, beta_q(q) + se_q(q) + 0.06*yrng, '*', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 18, ...
            'FontWeight', 'bold');
    end
end

ylim([yBot - 0.12*yrng, yTop + 0.18*yrng]);
box off;

% ===================== save =====================
outDir = fileparts(outPDF_q);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

set(figQ, 'Renderer', 'painters');
print(figQ, outPDF_q, '-dpdf', '-painters');
fprintf('✓ Saved quarter bar plot: %s\n', outPDF_q);

% ===================== result table =====================
quarterTbl = table((1:4)', qLabels(:), beta_q(:), se_q(:), p_q(:), n_q(:), ...
    'VariableNames', {'QuarterIdx','Quarter','Beta','SE','pValue','nBins'});

disp(quarterTbl);

end