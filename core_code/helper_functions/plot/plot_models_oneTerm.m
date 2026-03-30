function plot_models_oneTerm(Sel, models_to_plot, termName, t_norm, colSub, ...
                              AIC_mat, BIC_mat)
% Sel            : 1×nModels struct (同你现有的Sel)
% models_to_plot : e.g. [6, 2]，要画的model index
% termName       : e.g. 'b_{coh}'，只画这一个term
% t_norm         : 50×1 或 1×50，x轴
% colSub         : nSubj×3 颜色矩阵
% AIC_mat, BIC_mat : nBins×nModels (可选，传入才画winning shadow)

if nargin < 6
    AIC_mat = [];
    BIC_mat = [];
end

nPlot = numel(models_to_plot);
nSubj = size(Sel(1).beta_sub, 1);
x     = t_norm(:)';
nBins = numel(x);

lw_sub    = 0.55;
lw_pool   = 1.20;
alphaSub  = 0.10;
alphaPool = 0.10;

% 预计算winning bins（如果提供了AIC/BIC）
if ~isempty(AIC_mat) && ~isempty(BIC_mat)
    [~, bestAIC_idx] = min(AIC_mat, [], 2, 'omitnan');
    [~, bestBIC_idx] = min(BIC_mat, [], 2, 'omitnan');
    doWin = true;
else
    doWin = false;
end

figure('Color', 'w');

for p = 1:nPlot
    m  = models_to_plot(p);

    % 找term index
    termThis = termName{p};   % 每个subplot用自己的term
    tt = find(strcmp(Sel(m).termLabels, termThis), 1, 'first');
    title(ax, sprintf('Model %d — %s', m, termThis), 'Interpreter', 'none');

    subplot(nPlot, 1, p);
    ax = gca;
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'off');

    % ── winning shadow ──────────────────────────────────────
    if doWin
        aic_win = find(bestAIC_idx == m);
        bic_win = find(bestBIC_idx == m);
        for k = 1:numel(aic_win)
            xv = t_norm(aic_win(k));
            hw = (t_norm(2)-t_norm(1))/2;
            xregion(xv-hw, xv+hw, 'FaceColor', [0.5 0.5 0.5], ...
                'FaceAlpha', 0.20, 'HandleVisibility', 'off');
        end
        for k = 1:numel(bic_win)
            xv = t_norm(bic_win(k));
            hw = (t_norm(2)-t_norm(1))/2;
            xregion(xv-hw, xv+hw, 'FaceColor', 'r', ...
                'FaceAlpha', 0.15, 'HandleVisibility', 'off');
        end
    end

    yline(ax, 0, 'k--', 'LineWidth', 0.6, 'HandleVisibility', 'off');
    ylim(ax, [-0.25, 0.25]);

    if isempty(tt)
        title(ax, sprintf('Model %d — term "%s" not found', m, termName));
        continue;
    end

    beta_sub = Sel(m).beta_sub(:, :, tt);   % nSubj × nBins
    se_sub   = Sel(m).se_sub(:, :, tt);
    beta_pool = Sel(m).beta_pool(:, tt)';    % 1 × nBins
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

    % ── 格式 ────────────────────────────────────────────────
    xlim(ax, [x(1) x(end)]);
    xlabel(ax, 'Normalized time');
    ylabel(ax, 'beta value');
    title(ax, sprintf('Model %d — %s', m, termName), 'Interpreter', 'none');

    % legend只在第一个subplot放
    if p == 1
        hLeg = gobjects(nSubj+1, 1);
        legTxt = cell(nSubj+1, 1);
        for s = 1:nSubj
            hLeg(s) = plot(ax, nan, nan, '-', 'Color', colSub(s,:), 'LineWidth', 2);
            legTxt{s} = sprintf('Subject %d', s);
        end
        hLeg(nSubj+1) = plot(ax, nan, nan, 'k-', 'LineWidth', 2.5);
        legTxt{nSubj+1} = 'Pooled';
        legend(ax, hLeg, legTxt, 'Box', 'off', 'FontSize', 9, ...
            'Location', 'eastoutside');
    end
end

end