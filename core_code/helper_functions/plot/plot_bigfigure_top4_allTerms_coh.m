function plot_bigfigure_top4_allTerms_coh(Sel, t_norm, colSub, outPDF, termList, figTitle)

if nargin < 4 || isempty(outPDF)
    outPDF = 'BigFigure_Top4_CurrentFamily_AllTerms.pdf';
end
if nargin < 5
    termList = [];
end
if nargin < 6 || isempty(figTitle)
    figTitle = 'Top 4 models - current coh family';
end

nCols = numel(Sel);
if nCols == 0
    warning('Sel is empty. Nothing to plot.');
    return;
end

nSubj = size(Sel(1).beta_sub, 1);
x     = t_norm(:)';

if size(colSub,1) < nSubj
    colSub = lines(nSubj);
end

% ---------- collect terms ----------
allTerms = {};
for c = 1:nCols
    allTerms = [allTerms, Sel(c).termLabels]; %#ok<AGROW>
end
allTerms = unique(allTerms, 'stable');

if ~isempty(termList)
    keep = ismember(allTerms, termList);
    allTerms = allTerms(keep);
end

keep2 = false(size(allTerms));
for r = 1:numel(allTerms)
    term = allTerms{r};
    hasAny = false;
    for c = 1:nCols
        if any(strcmp(Sel(c).termLabels, term))
            hasAny = true;
            break;
        end
    end
    keep2(r) = hasAny;
end
allTerms = allTerms(keep2);

nRows = numel(allTerms);
if nRows == 0
    warning('No terms to plot.');
    return;
end

% ---------- row-specific y limits ----------
yLimByRow = cell(nRows,1);
for r = 1:nRows
    yLimByRow{r} = local_term_ylim_allModels_coh(Sel, allTerms{r});
end

% ---------- layout ----------
tileSize = 130;
gapX     = 20;
gapY     = 14;
labelW   = 90;

outerL = 40;
outerR = 30;
outerT = 35;
outerB = 100;

figW = outerL + labelW + nCols*tileSize + (nCols-1)*gapX + outerR;
figH = outerT + nRows*tileSize + (nRows-1)*gapY + outerB;

fig = figure('Color','w','Units','points','Position',[60 60 figW figH]);

pt2nx = @(pt) pt / figW;
pt2ny = @(pt) pt / figH;

L = pt2nx(outerL);
R = pt2nx(outerR);
T = pt2ny(outerT);
B = pt2ny(outerB);

gapXNorm   = pt2nx(gapX);
gapYNorm   = pt2ny(gapY);
labelWNorm = pt2nx(labelW);
tileWNorm  = pt2nx(tileSize);
tileHNorm  = pt2ny(tileSize);

usedW = labelWNorm + nCols*tileWNorm + (nCols-1)*gapXNorm;
usedH = nRows*tileHNorm + (nRows-1)*gapYNorm;

x0   = L + ((1 - L - R) - usedW)/2;
yTop = 1 - T - ((1 - T - B) - usedH)/2;

fontPanel = 9;
lw_sub    = 0.55;
lw_pool   = 1.20;
alphaSub  = 0.10;
alphaPool = 0.10;

lastAxPerCol = gobjects(1, nCols);
legendPlaced = false;

for r = 1:nRows
    yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

    % row label
    axLab = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', [x0, yPos, labelWNorm, tileHNorm]);
    axis(axLab, 'off');
    text(axLab, 0.50, 0.50, term_to_tex_compact_coh(allTerms{r}), ...
        'FontSize', 16, ...
        'Rotation', 90, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'Interpreter', 'latex', ...
        'Clipping', 'on');

    for c = 1:nCols
        xPos = x0 + labelWNorm + (c-1)*(tileWNorm + gapXNorm);

        ax = axes('Parent', fig, 'Units', 'normalized', ...
            'Position', [xPos, yPos, tileWNorm, tileHNorm]);
        hold(ax, 'on');
        grid(ax, 'on');
        box(ax, 'off');

        if r == 1
            title(ax, strrep(Sel(c).mName, '_', '\_'), ...
                'Interpreter', 'tex', ...
                'FontSize', 10, ...
                'FontWeight', 'bold');
        end

        tt = find(strcmp(Sel(c).termLabels, allTerms{r}), 1, 'first');

        if isempty(tt)
            axis(ax, 'off');
            continue;
        end

        xlim(ax, [0 1]);
        xticks(ax, 0:0.2:1);
        set(ax, 'FontSize', fontPanel, 'LineWidth', 0.8);
        yline(ax, 0, 'k--', 'LineWidth', 0.6, 'HandleVisibility', 'off');

        if r < nRows
            set(ax, 'XTickLabel', []);
        end
        lastAxPerCol(c) = ax;

        beta_sub = Sel(c).beta_sub(:,:,tt);
        se_sub   = Sel(c).se_sub(:,:,tt);

        % ---------- subject lines ----------
        hSub = gobjects(nSubj,1);
        for s = 1:nSubj
            yv = squeeze(beta_sub(s,:));
            ev = squeeze(se_sub(s,:));

            ok = ~isnan(yv) & ~isnan(ev);
            if sum(ok) >= 2
                xx = x(ok);
                yy = yv(ok);
                ee = ev(ok);
                fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                    'EdgeColor', 'none', ...
                    'FaceAlpha', alphaSub, ...
                    'HandleVisibility', 'off');
            end

            hSub(s) = plot(ax, x, yv, '-', ...
                'Color', colSub(s,:), ...
                'LineWidth', lw_sub, ...
                'HandleVisibility', 'off');
        end

        % ---------- pooled line ----------
        yPool = Sel(c).beta_pool(:,tt)';
        ePool = Sel(c).se_pool(:,tt)';

        okp = ~isnan(yPool) & ~isnan(ePool);
        if sum(okp) >= 2
            xx = x(okp);
            ym = yPool(okp);
            es = ePool(okp);
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor', 'none', ...
                'FaceAlpha', alphaPool, ...
                'HandleVisibility', 'off');
        end

        hPool = plot(ax, x, yPool, 'k-', ...
            'LineWidth', lw_pool, ...
            'HandleVisibility', 'off');

        ylim(ax, yLimByRow{r});

        % ---------- legend once ----------
        if ~legendPlaced
            hLeg = gobjects(nSubj+1,1);
            legText = cell(nSubj+1,1);

            for s = 1:nSubj
                hLeg(s) = plot(ax, nan, nan, '-', ...
                    'Color', colSub(s,:), 'LineWidth', 2.0);
                legText{s} = sprintf('Subject %d', s);
            end

            hLeg(nSubj+1) = plot(ax, nan, nan, 'k-', 'LineWidth', 2.5);
            legText{nSubj+1} = 'Pooled';

            legend(ax, hLeg, legText, ...
                'Box', 'off', ...
                'FontSize', 9, ...
                'Location', 'northwest');

            legendPlaced = true;
        end
    end
end

for c = 1:nCols
    ax = lastAxPerCol(c);
    if ~isempty(ax) && isgraphics(ax) && strcmp(get(ax,'Visible'),'on')
        xlabel(ax, 'Normalized time (0--1)', 'Interpreter', 'latex', 'FontSize', 10);
    end
end

if exist('sgtitle', 'file') == 2
    sgtitle(figTitle, 'FontWeight', 'bold', 'Interpreter', 'none');
else
    annotation(fig, 'textbox', [0 0.97 1 0.03], ...
        'String', figTitle, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold');
end

set(fig, 'Renderer', 'painters');
set(fig, 'PaperUnits', 'points');
set(fig, 'PaperSize', [figW figH]);
set(fig, 'PaperPosition', [0 0 figW figH]);
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperOrientation', 'portrait');

print(fig, outPDF, '-dpdf', '-painters');
fprintf('✓ Saved: %s\n', outPDF);

end

% ============================================================
function yLim = local_term_ylim_allModels_coh(Sel, termLabel)

vals = [];

for c = 1:numel(Sel)
    tt = find(strcmp(Sel(c).termLabels, termLabel), 1, 'first');
    if isempty(tt)
        continue;
    end

    bSub = Sel(c).beta_sub(:,:,tt);
    eSub = Sel(c).se_sub(:,:,tt);

    bPool = Sel(c).beta_pool(:,tt);
    ePool = Sel(c).se_pool(:,tt);

    vals = [vals; ...
            bSub(:); bSub(:)+eSub(:); bSub(:)-eSub(:); ...
            bPool(:); bPool(:)+ePool(:); bPool(:)-ePool(:)]; %#ok<AGROW>
end

vals = vals(~isnan(vals));

if isempty(vals)
    yLim = [-1 1];
    return;
end

lo = prctile(vals, 2);
hi = prctile(vals, 98);

if abs(hi - lo) < 1e-6
    lo = lo - 1;
    hi = hi + 1;
end

pad = 0.10 * (hi - lo + eps);
yLim = [lo - pad, hi + pad];

end

% ============================================================
function out = term_to_tex_compact_coh(lbl)

switch lbl
    case 'b0 (Intercept)'
        out = '$b_0$';

    case 'b_{corr}'
        out = '$b_{\mathrm{corr}}$';

    case 'b_{rt}'
        out = '$b_{\mathrm{rt}}$';

    case 'b_{coh}'
        out = '$b_{\mathrm{coh}}$';

    case 'b_{perf}'
        out = '$b_{\mathrm{perf}}$';

    case 'b_{vol}'
        out = '$b_{\mathrm{vol}}$';

    case 'b_{perf×vol}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}}$';

    case 'b_{perf×vol×coh}'
        out = '$b_{\mathrm{perf}\times\mathrm{vol}\times\mathrm{coh}}$';

    otherwise
        out = ['$' lbl '$'];
end

end