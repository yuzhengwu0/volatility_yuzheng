function resVol_mat = compute_resVol_time(motion_energy, nBins, winLen, tol)
% Compute residual volatility across all trials

nTrials = numel(motion_energy);
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

    nWin  = nFrames - winLen + 1; % how many windows we can get
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

t_norm = linspace(0, 1, nBins); % do timebin

MEAN_norm = nan(nTrials, nBins); % save mean alltrials x alltimebin
STD_norm  = nan(nTrials, nBins); % save std alltrials x alltimebin

for tr = 1:nTrials
    mu_tr = evidence_strength{tr};
    sd_tr = volatility_strength{tr};
    if isempty(mu_tr) || isempty(sd_tr), continue; end

    % avoid error
    nWin_tr = min(numel(mu_tr), numel(sd_tr)); 
    mu_tr   = mu_tr(1:nWin_tr);
    sd_tr   = sd_tr(1:nWin_tr);

    % do linear interpolation within 0-1, time normalization
    t_orig = linspace(0, 1, nWin_tr);
    MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
    STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
end

% calculate residual (remove mean effect) vol for each time bin
resVol_mat = nan(size(STD_norm));

for b = 1:nBins
    y  = STD_norm(:, b);
    x1 = abs(MEAN_norm(:, b));

    mask_b = ~isnan(y) & ~isnan(x1);
    if sum(mask_b) < 3
        continue;
    end

    % set a regression here, remove the part of volatility can be
    % explained by strength
    % --> y (motion energy) ~ beta0 + beta1 * evidence strength
    Xb    = [ones(sum(mask_b),1), x1(mask_b)];
    y_use = y(mask_b);

    beta  = Xb \ y_use;
    resid = y_use - Xb * beta; % raw vol - the part can be explained

    % save in nTrials x bins format
    tmpv = nan(size(y));
    tmpv(mask_b) = resid;
    resVol_mat(:, b) = tmpv;
end
end