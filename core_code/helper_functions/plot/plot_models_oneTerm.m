function plot_models_oneTerm(Sel, models_to_plot, termName, t_norm, colSub, ...
                              AIC_mat, BIC_mat)
% Plot one beta term per subplot across specified models.
% Sel            : 1×nModels struct with fields beta_sub, se_sub, beta_pool, se_pool, termLabels, mName
% models_to_plot : vector of Sel indices to plot, e.g. [1, 4]
% termName       : cell array of term strings, one per subplot, e.g. {'b_{vol}', 'b_{coh}'}
% t_norm         : normalized time vector (0~1), length nBins
% colSub         : nSubj×3 RGB color matrix for individual subjects
% AIC_mat        : nBins×nModels matrix of AIC values (optional)
% BIC_mat        : nBins×nModels matrix of BIC values (optional)

% ── default: skip winning shadows if AIC/BIC not provided ───
if nargin < 6
    AIC_mat = [];
    BIC_mat = [];
end

% ── basic dimensions ────────────────────────────────────────
nPlot = numel(models_to_plot);          % number of subplots
nSubj = size(Sel(1).beta_sub, 1);       % number of subjects
x     = t_norm(:)';                     % ensure row vector for plotting

% ── line and shading style parameters ───────────────────────
lw_sub    = 0.55;    % line width for individual subject traces
lw_pool   = 1.20;    % line width for pooled trace
alphaSub  = 0.10;    % transparency of per-subject SE shading
alphaPool = 0.10;    % transparency of pooled SE shading

% ── colors for AIC/BIC winning-bin shadows ──────────────────
colAIC = [0.85 0.85 0.85];   % light gray for AIC-winning bins
colBIC = [1.0  0.75 0.75];   % light pink for BIC-winning bins

% ── precompute which bin each model wins under AIC and BIC ──
if ~isempty(AIC_mat) && ~isempty(BIC_mat)
    [~, bestAIC_idx] = min(AIC_mat, [], 2, 'omitnan');  % nBins×1, value = winning column (1~nModels)
    [~, bestBIC_idx] = min(BIC_mat, [], 2, 'omitnan');
    doWin = true;
else
    doWin = false;
end

figure('Color', 'w');

for p = 1:nPlot
    m        = models_to_plot(p);   % Sel index for this subplot
    termThis = termName{p};         % term string to plot in this subplot

    % ── parse model number from mName to get AIC/BIC column ─
    % mName format: 'M0_base', 'M2_V', 'M4_PxV', etc.
    % M0 → column 1, M1 → column 2, ..., M5 → column 6
    mName = Sel(m).mName;
    tok   = regexp(mName, '^M(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        mCol = str2double(tok{1}) + 1;
    else
        mCol = m;   % fallback: assume Sel index equals AIC/BIC column
        warning('Cannot parse model index from mName "%s", using Sel index %d', mName, m);
    end

    % ── create subplot and axes ──────────────────────────────
    subplot(nPlot, 1, p);
    ax = gca;
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'off');

    % ── find this term's index in the model's term list ─────
    tt = find(strcmp(Sel(m).termLabels, termThis), 1, 'first');

    % ── draw AIC/BIC winning-bin background shading ─────────
    if doWin
        aic_win = find(bestAIC_idx == mCol);   % bins where this model wins AIC
        bic_win = find(bestBIC_idx == mCol);   % bins where this model wins BIC
        for k = 1:numel(aic_win)
            xv = t_norm(aic_win(k));
            hw = (t_norm(2) - t_norm(1)) / 2;  % half bin width in normalized time
            xregion(xv-hw, xv+hw, 'FaceColor', colAIC, ...
                'FaceAlpha', 0.25, 'HandleVisibility', 'off');
        end
        for k = 1:numel(bic_win)
            xv = t_norm(bic_win(k));
            hw = (t_norm(2) - t_norm(1)) / 2;
            xregion(xv-hw, xv+hw, 'FaceColor', colBIC, ...
                'FaceAlpha', 0.20, 'HandleVisibility', 'off');
        end
    end

    % ── horizontal zero reference line ──────────────────────
    yline(ax, 0, 'k--', 'LineWidth', 0.6, 'HandleVisibility', 'off');

    % ── skip data plotting if term not found in this model ──
    if isempty(tt)
        title(ax, sprintf('[%s]  term not found in %s', termThis, mName), ...
            'Interpreter', 'none', 'FontSize', 9);
        xlim(ax, [x(1) x(end)]);
        continue;
    end

    % ── extract beta and SE for this term ───────────────────
    beta_sub  = Sel(m).beta_sub(:, :, tt);   % nSubj × nBins
    se_sub    = Sel(m).se_sub(:, :, tt);
    beta_pool = Sel(m).beta_pool(:, tt)';    % 1 × nBins
    se_pool   = Sel(m).se_pool(:, tt)';

    % ── plot per-subject SE shading and trace ────────────────
    for s = 1:nSubj
        yv = beta_sub(s, :);
        ev = se_sub(s, :);
        ok = ~isnan(yv) & ~isnan(ev);   % skip NaN bins
        if sum(ok) >= 2
            xx = x(ok); yy = yv(ok); ee = ev(ok);
            fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                'EdgeColor', 'none', 'FaceAlpha', alphaSub, ...
                'HandleVisibility', 'off');
        end
        plot(ax, x, yv, '-', 'Color', colSub(s,:), 'LineWidth', lw_sub, ...
            'HandleVisibility', 'off');
    end

    % ── plot pooled SE shading and trace ────────────────────
    okp = ~isnan(beta_pool) & ~isnan(se_pool);
    if sum(okp) >= 2
        xx = x(okp); ym = beta_pool(okp); es = se_pool(okp);
        fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
            'EdgeColor', 'none', 'FaceAlpha', alphaPool, ...
            'HandleVisibility', 'off');
    end
    plot(ax, x, beta_pool, 'k-', 'LineWidth', lw_pool, 'HandleVisibility', 'off');

    % ── title: list all terms, highlight current with [ ] ───
    allTerms  = Sel(m).termLabels;
    termParts = cell(1, numel(allTerms));
    for k = 1:numel(allTerms)
        if strcmp(allTerms{k}, termThis)
            termParts{k} = ['[' allTerms{k} ']'];   % bracket the plotted term
        else
            termParts{k} = allTerms{k};
        end
    end
    titleStr = [mName '  |  ' strjoin(termParts, ',  ')];
    title(ax, titleStr, 'Interpreter', 'none', 'FontSize', 9);

    % ── axis formatting ──────────────────────────────────────
    xlim(ax, [x(1) x(end)]);
    ylim(ax, [-0.25, 0.25]);
    xlabel(ax, 'Normalized time');
    ylabel(ax, 'beta value');

    % ── legend: subjects + pooled + AIC/BIC shadow swatches ─
    nLeg   = nSubj + 1 + 2 * doWin;
    hLeg   = gobjects(nLeg, 1);
    legTxt = cell(nLeg, 1);
    for s = 1:nSubj
        hLeg(s)   = plot(ax, nan, nan, '-', 'Color', colSub(s,:), 'LineWidth', 2);
        legTxt{s} = sprintf('Subject %d', s);
    end
    hLeg(nSubj+1)   = plot(ax, nan, nan, 'k-', 'LineWidth', 2.5);
    legTxt{nSubj+1} = 'Pooled';
    if doWin
        hLeg(nSubj+2) = fill(ax, nan, nan, colAIC, 'EdgeColor', 'none', 'FaceAlpha', 0.18);
        legTxt{nSubj+2} = 'AIC win';
        hLeg(nSubj+3) = fill(ax, nan, nan, colBIC, 'EdgeColor', 'none', 'FaceAlpha', 0.18);
        legTxt{nSubj+3} = 'BIC win';
    end
    legend(ax, hLeg, legTxt, 'Box', 'off', 'FontSize', 9, 'Location', 'eastoutside');
end

end