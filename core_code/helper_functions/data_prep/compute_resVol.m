function [resVol_mat, resVol, evidence_strength, volatility_strength] = compute_resVol(motion_energy, vol, nBins, winLen, tol, coh, req)
% Compute residual volatility across all trials
% Also recode volatility into cond:
%   low vol  -> cond = 1
%   high vol -> cond = 2
%
% Outputs:
%   resVol_mat  : raw residual volatility (trial x time bin)
%   resVol_time : z-scored residual volatility across all trials and bins
%   cond        : recoded volatility condition

nTrials = numel(motion_energy);

if numel(vol) ~= nTrials
    error('vol must have the same number of trials as motion_energy.');
end

%% ===================== Compute evidence / volatility strength =====================
mask = req == 2 & coh ~= 0;

for tr = find(mask(:))'
    motion_energy{tr} = -1 * motion_energy{tr};
end

nWin_fixed = 36 - winLen + 1;
evidence_strength   = nan(nTrials, nWin_fixed);
volatility_strength = nan(nTrials, nWin_fixed);

for tr = 1:nTrials
    frames = motion_energy{tr};
    trace  = frames(:)';
    % last_nz = 38;
    
    % if isempty(last_nz)
    %     continue;
    % end
    
    % only crop bin 3-38
    trace_eff = trace(3:38);
    nFrames   = numel(trace_eff);
    nWin      = nFrames - winLen + 1;
    
    if nWin ~= nWin_fixed
        fprintf('Warning: trial %d has unexpected nWin = %d\n', tr, nWin);
        continue;
    end
    
    m_win = nan(1, nWin);
    s_win = nan(1, nWin);
    
    % get mean and std(raw vol) in each window
    for w = 1:nWin
        seg      = trace_eff(w : w + winLen - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end
    
    evidence_strength(tr, :)   = m_win;
    volatility_strength(tr, :) = s_win;
end


%% ===================== Residual volatility =====================
x_all = evidence_strength(:);
% x_all = abs(evidence_strength(:));
y_all = volatility_strength(:);

mask_all = ~isnan(x_all) & ~isnan(y_all);

x_use = x_all(mask_all);
y_use = y_all(mask_all);

Xall = [ones(sum(mask_all),1), x_use];
beta_all = Xall \ y_use;

yhat_all = Xall * beta_all;
resid_all = y_use - yhat_all;

tmp_all = nan(size(x_all));
tmp_all(mask_all) = resid_all;

resVol_mat = reshape(tmp_all, size(evidence_strength));

% test scatter plot
figure
scatter(x_use, y_use, 5, 'filled')
hold on
x_line = linspace(min(x_use), max(x_use), 200)';
y_line = beta_all(1) + beta_all(2) * x_line;

plot(x_line, y_line, 'r-', 'LineWidth', 2);

%% ===================== Z-score residual volatility =====================
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');

if sd_all == 0 || isnan(sd_all)
    resVol = zeros(size(resVol_mat));
else
    resVol = (resVol_mat - mu_all) ./ sd_all;
end


end
