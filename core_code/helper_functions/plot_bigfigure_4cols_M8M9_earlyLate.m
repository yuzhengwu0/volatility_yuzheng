function plot_bigfigure_4cols_M8M9_earlyLate(SelEarly, SelLate, t_norm, colSub, outPDF, termList, figTitle)
% 4 columns:
%   1) M8 EARLY
%   2) M8 LATE
%   3) M9 EARLY
%   4) M9 LATE
%
% y-axis rule (NEW):
%   each ROW (same term) shares one y-range across all 4 panels
%   not global across rows
%   default uses beta range only (not beta +/- se)

if nargin < 5 || isempty(outPDF)
    outPDF = 'BigFigure_M8M9_EarlyLate_AllTerms.pdf';
end
if nargin < 6
    termList = [];
end
if nargin < 7
    figTitle = 'EARLY vs LATE | M8 vs M9';
end

% ===================== USER-ADJUSTABLE Y-LIM SETTINGS =====================
ROW_YLIM_MODE      = 'beta_se';   % 'beta' or 'beta_se'
ROW_YLIM_PAD_FRAC  = 0.15;     % extra padding fraction
ROW_YLIM_MIN_HALF  = 0.03;     % minimum half-range, avoids over-flat axes

% optional visual settings
DRAW_ZERO_LINE = true;
lw_sub    = 0.9;
lw_mean   = 1.2;
alphaSub  = 0.10;
alphaMean = 0.08;
fontPanel = 9;

SHOW_SUBJECT_SE = true;   % 是否画每个subject自己的SE色块
SHOW_MEAN_SE    = true;    % 是否画黑线(mean)的SEM色块

nameE = string({SelEarly.mName});
nameL = string({SelLate.mName});

iM8E = find(contains(nameE, "M8"), 1, 'first');
iM9E = find(contains(nameE, "M9"), 1, 'first');
iM8L = find(contains(nameL, "M8"), 1, 'first');
iM9L = find(contains(nameL, "M9"), 1, 'first');

if any(isempty([iM8E iM9E iM8L iM9L]))
    error('Did not find M8/M9 in SelEarly/SelLate. Check mName strings.');
end

Panels(1).Sel = SelEarly(iM8E); Panels(1).title = 'M8 EARLY';
Panels(2).Sel = SelLate(iM8L);  Panels(2).title = 'M8 LATE';
Panels(3).Sel = SelEarly(iM9E); Panels(3).title = 'M9 EARLY';
Panels(4).Sel = SelLate(iM9L);  Panels(4).title = 'M9 LATE';

nCols = 4;
nSubj = size(Panels(1).Sel.beta_sub,1);
x     = t_norm(:)';

% ---- union of terms ----
allTerms = {};
for c = 1:nCols
    allTerms = [allTerms, Panels(c).Sel.termLabels]; %#ok<AGROW>
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
        if any(strcmp(Panels(c).Sel.termLabels, term))
            hasAny = true;
            break;
        end
    end
    keep2(r) = hasAny;
end
allTerms = allTerms(keep2);

nRows = numel(allTerms);
if nRows == 0
    warning('No terms to plot. Skipping: %s', outPDF);
    return;
end

% ---- NEW: y-lims per ROW, shared across all 4 panels ----
rowYLim = cell(nRows,1);
for r = 1:nRows
    rowYLim{r} = local_row_ylim_4panels( ...
        Panels, allTerms{r}, ...
        ROW_YLIM_MODE, ROW_YLIM_PAD_FRAC, ROW_YLIM_MIN_HALF);
end

% ===================== Figure layout =====================
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
set(fig, 'Name', 'EARLY/LATE combined: M8 vs M9');

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

lastAxPerCol = gobjects(1,nCols);

for r = 1:nRows
    yPos = yTop - r*tileHNorm - (r-1)*gapYNorm;

    % left label
    axLab = axes('Parent',fig,'Units','normalized','Position',[x0, yPos, labelWNorm, tileHNorm]);
    axis(axLab,'off');
    text(axLab, 0.50, 0.50, term_to_tex_compact(allTerms{r}), ...
        'FontSize', 10, 'FontWeight','bold', ...
        'Rotation', 90, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'Interpreter','tex', ...
        'Clipping','on');

    for c = 1:nCols
        xPos = x0 + labelWNorm + (c-1)*(tileWNorm + gapXNorm);
        ax = axes('Parent',fig,'Units','normalized','Position',[xPos, yPos, tileWNorm, tileHNorm]);
        hold(ax,'on'); grid(ax,'on'); box(ax,'off');

        if r == 1
            ttl = strrep(Panels(c).title,'_','\_');
            title(ax, ttl, 'Interpreter','tex','FontSize',11,'FontWeight','bold');
        end

        xlim(ax,[0 1]);
        xticks(ax,0:0.2:1);
        set(ax,'FontSize',fontPanel,'LineWidth',0.8);

        if DRAW_ZERO_LINE
            yline(ax,0,'k--','LineWidth',0.6,'HandleVisibility','off');
        end

        if r < nRows
            set(ax,'XTickLabel',[]);
        end
        lastAxPerCol(c) = ax;

        termListHere = Panels(c).Sel.termLabels;
        tt = find(strcmp(termListHere, allTerms{r}), 1, 'first');

        % same y-lim for entire row
        ylim(ax, rowYLim{r});

        if isempty(tt)
            text(ax, 0.5, 0.5, '—', 'Units','normalized', ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontSize', 16, 'Color', [0.35 0.35 0.35]);
            continue;
        end

        beta_sub = Panels(c).Sel.beta_sub(:,:,tt);
        se_sub   = Panels(c).Sel.se_sub(:,:,tt);

        % subject curves + subject SE band
        for s = 1:nSubj
            yv = squeeze(beta_sub(s,:));
            ev = squeeze(se_sub(s,:));

            ok = isfinite(yv) & isfinite(ev);
            if sum(ok) >= 2
                xx = x(ok);
                yy = yv(ok);
                ee = ev(ok);

                if SHOW_SUBJECT_SE
                    fill(ax, [xx fliplr(xx)], [yy-ee fliplr(yy+ee)], colSub(s,:), ...
                        'EdgeColor','none','FaceAlpha',alphaSub,'HandleVisibility','off');
                end
            end

            okLine = isfinite(yv);
            if sum(okLine) >= 2
                plot(ax, x(okLine), yv(okLine), '-', ...
                    'Color', colSub(s,:), 'LineWidth', lw_sub, 'HandleVisibility','off');
            end
        end

        % mean line + SEM
        yMean = mean(beta_sub,1,'omitnan');
        nEff  = sum(isfinite(beta_sub),1);
        ySEM  = std(beta_sub,0,1,'omitnan') ./ sqrt(max(nEff,1));

        okm = isfinite(yMean) & isfinite(ySEM);
        if SHOW_MEAN_SE && sum(okm) >= 2
            xx = x(okm);
            ym = yMean(okm);
            es = ySEM(okm);
        
            fill(ax, [xx fliplr(xx)], [ym-es fliplr(ym+es)], [0 0 0], ...
                'EdgeColor','none','FaceAlpha',alphaMean,'HandleVisibility','off');
        end

        okMeanLine = isfinite(yMean);
        if sum(okMeanLine) >= 2
            plot(ax, x(okMeanLine), yMean(okMeanLine), 'k-', ...
                'LineWidth', lw_mean, 'HandleVisibility','off');
        end
    end
end

for c = 1:nCols
    ax = lastAxPerCol(c);
    if ~isempty(ax) && isgraphics(ax)
        xlabel(ax,'Normalized time (0--1)','Interpreter','latex','FontSize',10);
    end
end

% legend
axLeg = axes('Parent',fig,'Units','normalized','Position',[0.10 0.01 0.35 0.07]);
axis(axLeg,'off'); hold(axLeg,'on');

hLeg = gobjects(nSubj+1,1);
legText = cell(nSubj+1,1);

for s = 1:nSubj
    hLeg(s) = plot(axLeg, nan, nan, '-', 'LineWidth', 2.5, 'Color', colSub(s,:));
    legText{s} = sprintf('Subject %d', s);
end
hLeg(nSubj+1) = plot(axLeg, nan, nan, 'k-', 'LineWidth', 3.0);
legText{nSubj+1} = 'Mean';

lgd = legend(axLeg, hLeg, legText, 'Orientation','vertical', 'Location','northwest');
lgd.Box = 'off';
lgd.FontSize = 11;

if exist('sgtitle','file') == 2
    sgtitle(figTitle, 'FontWeight','bold');
end

% export
set(fig,'Renderer','painters');
set(fig,'PaperUnits','points');
set(fig,'PaperSize',[figW figH]);
set(fig,'PaperPosition',[0 0 figW figH]);
set(fig,'PaperPositionMode','manual');
set(fig,'PaperOrientation','portrait');

print(fig, outPDF, '-dpdf', '-painters');
fprintf('✓ Saved: %s\n', outPDF);

end