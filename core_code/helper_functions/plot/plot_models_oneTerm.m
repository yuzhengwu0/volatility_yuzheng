function plot_models_oneTerm(Sel, models_to_plot, termName, t_norm, colSub, ...
                              AIC_mat, BIC_mat)

if nargin < 6
    AIC_mat = [];
    BIC_mat = [];
end

nPlot = numel(models_to_plot);
nSubj = size(Sel(1).beta_sub, 1);
x     = t_norm(:)';

lw_sub    = 0.55;
lw_pool   = 1.20;
alphaSub  = 0.10;
alphaPool = 0.10;

colAIC = [0.85 0.85 0.85];   % 很浅的灰
colBIC = [1.0 0.75 0.75];    % 很浅的粉红

if ~isempty(AIC_mat) && ~isempty(BIC_mat)
    [~, bestAIC_idx] = min(AIC_mat, [], 2, 'omitnan');  % nBins×1，值是1~6
    [~, bestBIC_idx] = min(BIC_mat, [], 2, 'omitnan');
    doWin = true;
else
    doWin = false;
end

figure('Color', 'w');

for p = 1:nPlot
    m        = models_to_plot(p);
    termThis = termName{p};

    % ── 从 mName 解析 AIC/BIC 列index ───────────────────────
    % mName 格式如 'M4_PxV'，取第一个数字
    mName = Sel(m).mName;
    tok   = regexp(mName, '^M(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        mCol = str2double(tok{1}) + 1;  % M0→1, M1→2, M2→3, ... M5→6
    else
        mCol = m;  % fallback
        warning('Cannot parse model index from mName "%s", using Sel index %d', mName, m);
    end

    subplot(nPlot, 1, p);
    ax = gca;
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'off');

    tt = find(strcmp(Sel(m).termLabels, termThis), 1, 'first');

    % ── winning shadow ──────────────────────────────────────
    if doWin
        aic_win = find(bestAIC_idx == mCol);
        bic_win = find(bestBIC_idx == mCol);
        for k = 1:numel(aic_win)
            xv = t_norm(aic_win(k));
            hw = (t_norm(2) - t_norm(1)) / 2;
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

    yline(ax, 0, 'k--', 'LineWidth', 0.6, 'HandleVisibility', 'off');

    if isempty(tt)
        title(ax, sprintf('[%s]  term not found in %s', termThis, mName), ...
            'Interpreter', 'none', 'FontSize', 9);
        xlim(ax, [x(1) x(end)]);
        continue;
    end

    beta_sub  = Sel(m).beta_sub(:, :, tt);
    se_sub    = Sel(m).se_sub(:, :, tt);
    beta_pool = Sel(m).beta_pool(:, tt)';
    se_pool   = Sel(m).se_pool(:, tt)';

    % ── per subject ─────────────────────────────────────────
    for s = 1:nSubj
        yv = beta_sub(s, :);
        ev = se_sub(s, :);
        ok = ~isnan(yv) & ~isnan(ev);
        if sum(ok) >= 2
            xx = x(ok); yy = yv(ok); ee = ev(ok);
            fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                'EdgeColor', 'none', 'FaceAlpha', alphaSub, ...
                'HandleVisibility', 'off');
        end
        plot(ax, x, yv, '-', 'Color', colSub(s,:), 'LineWidth', lw_sub, ...
            'HandleVisibility', 'off');
    end

    % ── pooled ──────────────────────────────────────────────
    okp = ~isnan(beta_pool) & ~isnan(se_pool);
    if sum(okp) >= 2
        xx = x(okp); ym = beta_pool(okp); es = se_pool(okp);
        fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
            'EdgeColor', 'none', 'FaceAlpha', alphaPool, ...
            'HandleVisibility', 'off');
    end
    plot(ax, x, beta_pool, 'k-', 'LineWidth', lw_pool, 'HandleVisibility', 'off');

    % ── title：列出所有term，当前term加方括号 ────────────────
    allTerms  = Sel(m).termLabels;
    termParts = cell(1, numel(allTerms));
    for k = 1:numel(allTerms)
        if strcmp(allTerms{k}, termThis)
            termParts{k} = ['[' allTerms{k} ']'];
        else
            termParts{k} = allTerms{k};
        end
    end
    titleStr = [mName '  |  ' strjoin(termParts, ',  ')];
    title(ax, titleStr, 'Interpreter', 'none', 'FontSize', 9);

    % ── 格式 ────────────────────────────────────────────────
    xlim(ax, [x(1) x(end)]);
    ylim(ax, [-0.25, 0.25]);
    xlabel(ax, 'Normalized time');
    ylabel(ax, 'beta value');

    % ── legend（每个subplot都放，eastoutside）────────────────
    nLeg  = nSubj + 1 + 2 * doWin;
    hLeg  = gobjects(nLeg, 1);
    legTxt = cell(nLeg, 1);
    for s = 1:nSubj
        hLeg(s)   = plot(ax, nan, nan, '-', 'Color', colSub(s,:), 'LineWidth', 2);
        legTxt{s} = sprintf('Subject %d', s);
    end
    hLeg(nSubj+1)   = plot(ax, nan, nan, 'k-', 'LineWidth', 2.5);
    legTxt{nSubj+1} = 'Pooled';
    if doWin
        hLeg(nSubj+2) = fill(ax, nan, nan, colAIC, 'EdgeColor','none','FaceAlpha',0.18);
        legTxt{nSubj+2} = 'AIC win';
        hLeg(nSubj+3) = fill(ax, nan, nan, colBIC, 'EdgeColor','none','FaceAlpha',0.18);
        legTxt{nSubj+3} = 'BIC win';
    end
    legend(ax, hLeg, legTxt, 'Box', 'off', 'FontSize', 9, 'Location', 'eastoutside');
end

end