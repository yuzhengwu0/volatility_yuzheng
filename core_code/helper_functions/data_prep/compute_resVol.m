function [resVol_mat, resVol, evidence_strength, volatility_strength] = compute_resVol(motion_energy, vol, nBins, winLen, tol, coh, cond, req)
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
zerocoh = isnan(req);
mask    = req == 2 & coh ~= 0;

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
%% ===================== Time normalization =====================
% t_norm = linspace(0, 1, nBins);
% 
% MEAN_norm = nan(nTrials, nBins);
% STD_norm  = nan(nTrials, nBins);
% 
% for tr = 1:nTrials
%     mu_tr = evidence_strength{tr};
%     sd_tr = volatility_strength{tr};
% 
%     if isempty(mu_tr) || isempty(sd_tr)
%         continue;
%     end
% 
%     nWin_tr = min(numel(mu_tr), numel(sd_tr)); 
%     mu_tr   = mu_tr(1:nWin_tr);
%     sd_tr   = sd_tr(1:nWin_tr);
% 
%     t_orig = linspace(0, 1, nWin_tr);
%     MEAN_norm(tr,:) = interp1(t_orig, mu_tr, t_norm, 'linear');
%     STD_norm(tr,:)  = interp1(t_orig, sd_tr, t_norm, 'linear');
% end

%% ===================== Residual volatility =====================
resVol_mat = nan(size(evidence_strength));
% evidence_strength(tr)   = median(m_win);
% volatility_strength(tr) = median(s_win);

nWin_fixed = size(evidence_strength, 2);

for b = 1:nWin_fixed
    x1  = evidence_strength(:, b);
    y = volatility_strength(:, b);

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

%% plotting for each trial
figure;
tiledlayout(3,1)
nexttile;
hold on
% motion energy mean for trial 1 - low vol
plot(evidence_strength(9,:))
% motion energy mean for trial 3 - high vol
plot(evidence_strength(8,:))
title("motion energy mean")
xlabel("window")
ylabel("evidnce strength")
legend({"low vol", "high vol"})

nexttile;
hold on
% motion energy STD for trial 1 - low vol
plot(volatility_strength(9,:))
% motion energy STD for trial 3 - high vol
plot(volatility_strength(8,:))
title("motion energy STD")
xlabel("window")
ylabel("volatility strength")
legend({"low vol", "high vol"})

nexttile;
hold on
% resVol for trial 1 - low vol
plot(resVol(9,:))
% resVol for trial 3 - high vol
plot(resVol(8,:))
title("resVol")
xlabel("window")
ylabel("resVol strength")
legend({"low vol", "high vol"})

%% nested plotting loop for each trial (very thin line) and mean, divided by volatility (diff colors) and coherence (diff plots)
% figure;
% tiledlayout(length(unique(coh)), 3)
% 
% thiscoh = unique(coh);
% for icoh = 1:numel(thiscoh)
% 
%     %evidence strength
%     nexttile;
%     hold on;
%     % cond == 1
%     idx1 = (coh == thiscoh(icoh)) & (cond == 1);
%     data1 = evidence_strength(idx1, :);
%     plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
%     mu1 = mean(data1, 1, 'omitnan');
%     % sd1 = std(data1, 0, 1, 'omitnan');
% 
%     % cond == 2
%     idx2 = (coh == thiscoh(icoh)) & (cond == 2);
%     data2 = evidence_strength(idx2, :);
%     plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
%     mu2 = mean(data2, 1, 'omitnan');
%     % sd2 = std(data2, 0, 1, 'omitnan');
% 
%     x = 1:size(evidence_strength, 2);
%     plot(x, mu1, 'b', 'LineWidth', 1.5)
%     plot(x, mu2, 'r', 'LineWidth', 1.5)
%     % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
%     % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);
% 
%     ylim([-0.0002 0.0005])
%     title(sprintf('coh = %g', thiscoh(icoh)))
%     xlabel('windows')
%     ylabel('evidence strength')
% 
%     %volatility strength
%     nexttile;
%     hold on;
%     % cond == 1
%     idx1 = (coh == thiscoh(icoh)) & (cond == 1);
%     data1 = volatility_strength(idx1, :);
%     plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
%     mu1 = mean(data1, 1, 'omitnan');
%     % sd1 = std(data1, 0, 1, 'omitnan');
% 
%     % cond == 2
%     idx2 = (coh == thiscoh(icoh)) & (cond == 2);
%     data2 = volatility_strength(idx2, :);
%     plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
%     mu2 = mean(data2, 1, 'omitnan');
%     % sd2 = std(data2, 0, 1, 'omitnan');
% 
%     x = 1:size(volatility_strength, 2);
%     plot(x, mu1, 'b', 'LineWidth', 1.5)
%     plot(x, mu2, 'r', 'LineWidth', 1.5);
%     % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
%     % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);
% 
%     ylim([0 0.0002])
%     title(sprintf('coh = %g', thiscoh(icoh)))
%     xlabel('windows')
%     ylabel('volatility strength')
% 
%     % resVol
%     nexttile;
%     hold on;
%     % cond == 1
%     idx1 = (coh == thiscoh(icoh)) & (cond == 1);
%     data1 = resVol(idx1, :);
%     plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
%     mu1 = mean(data1, 1, 'omitnan');
%     % sd1 = std(data1, 0, 1, 'omitnan');
% 
%     % cond == 2
%     idx2 = (coh == thiscoh(icoh)) & (cond == 2);
%     data2 = resVol(idx2, :);
%     plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
%     mu2 = mean(data2, 1, 'omitnan');
%     % sd2 = std(data2, 0, 1, 'omitnan');
% 
%     x = 1:size(resVol, 2);
%     plot(x, mu1, 'b', 'LineWidth', 1.5)
%     plot(x, mu2, 'r', 'LineWidth', 1.5)
%     % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
%     % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);
% 
%     ylim([-2.5 5])
%     title(sprintf('coh = %g', thiscoh(icoh)))
%     xlabel('windows')
%     ylabel('resVol')
% end
% h1 = plot(nan, nan, 'b', 'LineWidth', 2);
% h2 = plot(nan, nan, 'r', 'LineWidth', 2);
% legend([h1 h2], {'low vol', 'high vol'})
% 

end
