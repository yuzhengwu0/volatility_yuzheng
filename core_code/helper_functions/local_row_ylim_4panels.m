function yl = local_row_ylim_4panels(Panels, termLabel, mode, padFrac, minHalfSpan)
% one y-lim per ROW, shared across all 4 panels

vals = [];

for c = 1:numel(Panels)
    S = Panels(c).Sel;

    tt = find(strcmp(S.termLabels, termLabel), 1, 'first');
    if isempty(tt)
        continue;
    end

    b = squeeze(S.beta_sub(:,:,tt));

    if strcmpi(mode, 'beta_se')
        e = squeeze(S.se_sub(:,:,tt));
        vals = [vals; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
    else
        vals = [vals; b(:)]; %#ok<AGROW>
    end
end

vals = vals(isfinite(vals));

if isempty(vals)
    yl = [-1 1];
    return;
end

maxAbs = max(abs(vals));
halfSpan = max(maxAbs * (1 + padFrac), minHalfSpan);

yl = [-halfSpan, halfSpan];

end

function yLim = local_term_ylim_twoPanels(SelA, SelB, termLabel)
vals = [];

for S = {SelA, SelB}
    Sel = S{1};
    tt = find(strcmp(Sel.termLabels, termLabel), 1, 'first');
    if isempty(tt), continue; end

    b = Sel.beta_sub(:,:,tt);
    e = Sel.se_sub(:,:,tt);
    vals = [vals; b(:); b(:)+e(:); b(:)-e(:)]; %#ok<AGROW>
end

vals = vals(~isnan(vals));
if isempty(vals)
    yLim = [-1 1];
    return;
end

lo = prctile(vals, 2);
hi = prctile(vals, 98);

if abs(hi-lo) < 1e-6
    lo = lo - 1;
    hi = hi + 1;
end

pad = 0.10 * (hi - lo + eps);
yLim = [lo-pad, hi+pad];
end