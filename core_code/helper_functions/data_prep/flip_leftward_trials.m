%% change accuracy by ME

motion_energy = cfg.motion_energy;
motion_mat = cfg.motion_mat;
req = cfg.req;
coh = cfg.coh;
winLen = cfg.winLen;
vol = cfg.vol;

nTrials = numel(motion_energy);

if numel(vol) ~= nTrials
    error('vol must have the same number of trials as motion_energy.');
end

%% ===================== Compute evidence / volatility strength =====================
mask = req == 2;
motion_mat(mask, :) = -motion_mat(mask, :);

nWin_fixed = size(motion_mat, 2) - winLen + 1;
evidence_strength   = nan(nTrials, nWin_fixed);
volatility_strength = nan(nTrials, nWin_fixed);

for tr = 1:nTrials
    trace_eff = motion_mat(tr, :);
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


cfg.evidence_strength = evidence_strength;
cfg.volatility_strength = volatility_strength;
cfg.motion_mat = motion_mat;