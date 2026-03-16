%% metaD_vs_dprime_vol.m
% Goal:
%   For each subject and volatility level (low / high),
%   estimate d' and meta-d' with the type2 SDT toolbox,
%   then plot meta-d' vs d' (x = d', y = meta-d').

clear; clc; close all;

%% 0. Add type2 SDT toolbox to path
restoredefaultpath; rehash toolboxcache;
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/type2sdt'));

%% 1. Load your volatility dataset
data_path = 'all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

coh    = allStruct.rdm1_coh(:);        % coherence (not directly used here)
vol    = allStruct.rdm1_coh_std(:);    % volatility
resp   = allStruct.req_resp(:);        % 1 = right, 2 = left
correct= allStruct.correct(:);         % 1 = correct, 0 = incorrect
conf   = allStruct.confidence(:);      % 0–1 continuous confidence
subjID = allStruct.group(:);           % subject index (1/2/3)

% Remove NaN trials
valid = ~isnan(coh) & ~isnan(vol) & ~isnan(resp) & ...
        ~isnan(correct) & ~isnan(conf) & ~isnan(subjID);

coh    = coh(valid);
vol    = vol(valid);
resp   = resp(valid);
correct= correct(valid);
conf   = conf(valid);
subjID = subjID(valid);

%% 2. Build stim / response / rating in the format needed by the meta-d' toolbox

% 2.1 resp: 1/2 -> 0/1
resp01 = resp - 1;           % 1->0, 2->1

% 2.2 Use resp + correct to recover the true stim (0/1)
S01 = resp01;
wrong_idx = (correct == 0);
S01(wrong_idx) = 1 - resp01(wrong_idx);

% 2.3 meta-d' toolbox needs stim/resp = 1 or 2
stim12 = S01 + 1;            % 0/1 -> 1/2
resp12 = resp01 + 1;

% 2.4 Split 0–1 confidence into 4 bins, rating = 1..4
conf_clipped = conf;
conf_clipped(conf_clipped < 0) = 0;
conf_clipped(conf_clipped > 1) = 1;

edges = [0, 0.25, 0.5, 0.75, 1];          % four equal bins
rating = discretize(conf_clipped, edges, 'IncludedEdge', 'right');
rating(isnan(rating)) = 4;                % Should not happen; just for safety
R = 4;                                     % number of rating levels

% 2.5 Make data2 struct (same style as example data2)
data2 = struct();
data2.stim     = stim12(:);   % 1 or 2
data2.response = resp12(:);   % 1 or 2
data2.rating   = rating(:);   % 1..4
data2.vol      = vol(:);      % keep volatility for trial selection
data2.subjID   = subjID(:);   % subject index

%% 3. Set subjects and volatility conditions

subj_list = [1 2 3];
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('数据里的 volatility 水平不是两个，请检查！');
end
vol_low  = min(vol_levels);   % low volatility
vol_high = max(vol_levels);   % high volatility

nSub  = numel(subj_list);
nCond = 2;                    % 1 = low vol, 2 = high vol

dprime_mat   = nan(nSub, nCond);   % da
metad_mat    = nan(nSub, nCond);   % meta-da
Mratio_mat   = nan(nSub, nCond);   % M-ratio

%% 4. First define a small function: trials -> counts (nR_S1, nR_S2)
local_counts = @(data2, idx, R) local_trials2counts(data2, idx, R);

%% 5. Then define a function handle: fit meta-d'
%    Prefer MLE; if no Optimization toolbox, use SSE
useMLE = ~isempty(which('getIpOptions')) && ~isempty(which('fmincon'));
if useMLE
    fit_meta = @(n1,n2) fit_meta_d_MLE(n1,n2,1);   % s = 1
else
    fit_meta = @(n1,n2) fit_meta_d_SSE(n1,n2,1);
end

%% 6. Loop: run meta-d' fit for each subject × (low / high volatility)

for iSub = 1:nSub
    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Subject %d\n', thisSub);
    fprintf('=============================\n');

    % Trials for this subject
    idx_sub = (data2.subjID == thisSub);

    for iCond = 1:nCond

        if iCond == 1
            thisVol = vol_low;
            condName = 'low vol';
        else
            thisVol = vol_high;
            condName = 'high vol';
        end

        idx_vol = (data2.vol == thisVol);
        idx     = idx_sub & idx_vol;

        if sum(idx) < 50   % If trials are few, print a warning
            warning('Sub %d, %s: only %d trials. Fit may be unstable.', ...
                     thisSub, condName, sum(idx));
        end

        if sum(idx) == 0
            fprintf('Sub %d, %s: no trials, skip.\n', thisSub, condName);
            continue;
        end

        % 6.1 trials -> counts
        [nR_S1, nR_S2] = local_counts(data2, idx, R);

        % 6.2 Zero-count smoothing (same idea as example; avoid zeros)
        smooth_counts = @(v) v + (any(v==0) * (1/numel(v)));
        nR_S1 = smooth_counts(nR_S1);
        nR_S2 = smooth_counts(nR_S2);

        % 6.3 Fit meta-d'
        fit = fit_meta(nR_S1, nR_S2);

        dprime_mat(iSub, iCond) = fit.da;
        metad_mat(iSub, iCond)  = fit.meta_da;
        Mratio_mat(iSub, iCond) = fit.M_ratio;

        fprintf('%s: d'' = %.3f, meta-d'' = %.3f, M-ratio = %.3f (n=%d)\n', ...
                 condName, fit.da, fit.meta_da, fit.M_ratio, sum(idx));
    end
end


%% 7. Plot: x = d', y = meta-d', with different colors for low/high volatility

figure('Color','w'); hold on; box on; grid on;

% Low volatility points
hLow = scatter(dprime_mat(:,1), metad_mat(:,1), 80, ...
        'MarkerEdgeColor',[0 0.447 0.741], ...
        'MarkerFaceColor',[0.8 0.9 1], ...
        'DisplayName','low vol');

% High volatility points
hHigh = scatter(dprime_mat(:,2), metad_mat(:,2), 80, ...
        'MarkerEdgeColor',[0.850 0.325 0.098], ...
        'MarkerFaceColor',[1 0.8 0.7], ...
        'DisplayName','high vol');

% Add subject labels next to each point
for iSub = 1:nSub
    % low vol point for this subject
    xL = dprime_mat(iSub, 1);
    yL = metad_mat(iSub, 1);
    text(xL, yL, sprintf(' S%d', subj_list(iSub)), ...
         'Color',[0 0.447 0.741], 'FontSize', 10);

    % high vol point for this subject
    xH = dprime_mat(iSub, 2);
    yH = metad_mat(iSub, 2);
    text(xH, yH, sprintf(' S%d', subj_list(iSub)), ...
         'Color',[0.850 0.325 0.098], 'FontSize', 10);
end

% Diagonal line meta-d' = d'
xymax = max([dprime_mat(:); metad_mat(:)]) * 1.1;
plot([0 xymax], [0 xymax], 'k--', 'LineWidth', 1.5, ...
     'DisplayName','meta-d'' = d''');

xlabel('d'''); 
ylabel('meta-d''');
title('meta-d'' vs d'' by volatility (subjects 1–3)');
legend('Location','southeast');


%% 8. Helper: convert trials to SDT counts (nR_S1, nR_S2)
function [nR_S1, nR_S2] = local_trials2counts(data2, idx, R)

    % S1 / S2 labels
    isS1 = (data2.stim(idx) == 1);  isS1 = isS1(:);
    isS2 = ~isS1;

    % response
    resp = data2.response(idx);
    if iscell(resp), resp = cell2mat(resp); end
    if ~isvector(resp), resp = resp(:,1); end
    resp = resp(:);

    % rating
    rate = data2.rating(idx);
    if iscell(rate), rate = cell2mat(rate); end
    if ~isvector(rate), rate = rate(:,1); end
    rate = rate(:);

    % output count vectors
    nR_S1 = zeros(1, 2 * R);
    nR_S2 = zeros(1, 2 * R);

    % resp = 1 (S1), rating from high → low
    for r = 1:R
        rHL = R - r + 1;
        nR_S1(r) = sum(isS1 & resp==1 & rate==rHL);
        nR_S2(r) = sum(isS2 & resp==1 & rate==rHL);
    end

    % resp = 2 (S2), rating from low → high
    for r = 1:R
        nR_S1(R+r) = sum(isS1 & resp==2 & rate==r);
        nR_S2(R+r) = sum(isS2 & resp==2 & rate==r);
    end
end
