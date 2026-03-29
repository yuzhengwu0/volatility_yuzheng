function [resVol_mat, resVol, cond] = compute_resVol(motion_energy, vol, nBins, winLen, tol)
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

%% ===================== Recode volatility to cond =====================
vol_levels = unique(vol(~isnan(vol)));
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;

%% ===================== Compute evidence / volatility strength =====================
evidence_strength   = cell(nTrials, 1);
volatility_strength = cell(nTrials, 1);

for tr = 1:nTrials
    frames = motion_energy{tr};
    trace  = frames(:)'; 
    
    % because all the motion energy ends at different frames
    last_nz = find(abs(trace) > tol, 1, 'last');
    
    % skip the trial if ME is empty
    if isempty(last_nz)
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end
    
    % time normalization prep
    trace_eff = trace(1:last_nz);
    nFrames   = numel(trace_eff);

    % skip the trial if ME frame number less than window length
    if nFrames < winLen
        evidence_strength{tr}   = [];
        volatility_strength{tr} = [];
        continue;
    end

    nWin  = nFrames - winLen + 1;
    m_win = nan(1, nWin); 
    s_win = nan(1, nWin);

    % get mean and std(raw vol) in each window
    for w = 1:nWin
        seg      = trace_eff(w : w + winLen - 1);
        m_win(w) = mean(seg);
        s_win(w) = std(seg);
    end

    evidence_strength{tr}   = m_win;
    volatility_strength{tr} = s_win;
end

%% ===================== Time normalization =====================
t_norm = linspace(0, 1, nBins);

MEAN_norm = nan(nTrials, nBins);
STD_norm  = nan(nTrials, nBins);

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};

    if isempty(mu_tr) || isempty(sd_tr)
        continue;
    end

    nWin_tr = min(numel(mu_tr), numel(sd_tr)); 
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    t_orig = linspace(0, 1, nWin_tr);
    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

%% ===================== Residual volatility =====================
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3
        continue;
    end

    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb * beta;

    tmpv = nan(size(y));
    tmpv(mask_b) = resid;
    resVol_mat(:, b) = tmpv;
end

%% ===================== Z-score residual volatility =====================
mu_all = mean(resVol_mat(:), 'omitnan');
sd_all = std(resVol_mat(:),  'omitnan');

if sd_all == 0 || isnan(sd_all)
    resVol = zeros(size(resVol_mat));
else
    resVol = (resVol_mat - mu_all) ./ sd_all;
end

end