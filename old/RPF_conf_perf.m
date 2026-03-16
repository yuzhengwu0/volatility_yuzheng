%% RPF_perfConf_3subjects.m
% Use RPF toolbox to get:
%   P1(x) = d'(coherence)  → performance
%   P2(x) = p(high rating | coherence) → confidence
% And then R(P1) = P2, for subjects 1 / 2 / 3,
% with two conditions: low vs high volatility.

clear; clc;

%% 0. Add toolboxes to path
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/RPF-main'));
addpath(genpath('/Users/wuyuzheng/Documents/MATLAB/Palamedes1'));

RPF_check_toolboxes;   % Check that all needed toolboxes are found

%% 1. Load your data

data_path = 'all_with_me.mat';
tmp       = load(data_path, 'all');
allStruct = tmp.all;

% Basic fields
coh    = allStruct.rdm1_coh(:);        % coherence
resp   = allStruct.req_resp(:);        % 1 = right, 2 = left
correct= allStruct.correct(:);         % 1 = correct, 0 = incorrect
conf   = allStruct.confidence(:);      % 0–1 continuous confidence
vol    = allStruct.rdm1_coh_std(:);    % volatility
subjID = allStruct.group(:);           % subject index (1/2/3)

% Remove trials with any NaNs
valid = ~isnan(coh) & ~isnan(resp) & ~isnan(correct) & ...
        ~isnan(conf) & ~isnan(vol) & ~isnan(subjID);

coh    = coh(valid);
resp   = resp(valid);
correct= correct(valid);
conf   = conf(valid);
vol    = vol(valid);
subjID = subjID(valid);

% Map volatility to condition index
vol_levels = unique(vol);
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end

cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;   % low volatility
cond(vol == max(vol_levels)) = 2;   % high volatility

%% 2. Subject list
subj_list = [1 2 3];

for iSub = 1:numel(subj_list)

    thisSub = subj_list(iSub);
    fprintf('\n=============================\n');
    fprintf('Running RPF for subject %d\n', thisSub);
    fprintf('=============================\n');

    % Trials for this subject
    idxS = (subjID == thisSub);
    coh_s     = coh(idxS);
    resp_s    = resp(idxS);         % 1/2
    correct_s = correct(idxS);      % 0/1
    conf_s    = conf(idxS);
    cond_s    = cond(idxS);

    if isempty(coh_s)
        warning('Subject %d has no trials. Skipping.', thisSub);
        continue;
    end

    % 1) Turn responses 1/2 into 0/1 (RPF uses 0/1)
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % 2) Recover true stimulus S (0/1) from response + correct
    %    correct: S = response; wrong: S = 1 - response
    S01 = resp01;
    wrong_idx = (correct_s == 0);
    S01(wrong_idx) = 1 - resp01(wrong_idx);

    % ---------- Build trialData in the format RPF expects ----------

    nTr = numel(coh_s);  % number of trials for this subject

    % 1) Responses 1/2 → 0/1: 0 = say S1, 1 = say S2
    resp01 = resp_s - 1;   % 1 -> 0, 2 -> 1

    % 2) True stimulus:
    %    correct: stim = response; wrong: stim = 1 - response
    stim01 = resp01;
    wrong_idx = (correct_s == 0);
    stim01(wrong_idx) = 1 - resp01(wrong_idx);

    % 3) Map 0–1 continuous confidence to 4 rating levels (1–4)
    conf_clipped = conf_s;
    conf_clipped(conf_clipped < 0) = 0;
    conf_clipped(conf_clipped > 1) = 1;

    edges = [0, 0.25, 0.5, 0.75, 1];
    rating_s = discretize(conf_clipped, edges, 'IncludedEdge', 'right');
    rating_s(isnan(rating_s)) = 4;   % safety check
    nRatings = 4;

    % 4) Rename volatility condition for clarity
    condition_s = cond_s;   % 1 = low vol, 2 = high vol

    % 5) Build trialData (field names and shapes follow RPF example)

    trialData = struct();

    trialData.stimID    = stim01(:)';        % 1×nTr double, 0/1
    trialData.response  = resp01(:)';        % 1×nTr double, 0/1
    trialData.rating    = rating_s(:)';      % 1×nTr double, 1..4
    trialData.correct   = correct_s(:)';     % 1×nTr double, 0/1
    trialData.x         = coh_s(:)';         % 1×nTr double, coherence
    trialData.condition = condition_s(:)';   % 1×nTr double, 1/2
    trialData.RT        = nan(1, nTr);       % 1×nTr double, no RT → NaN
                                           
    %% 3. F1: P1(x) = d'(coherence)  (performance)

    F1 = struct();

    F1.info.DV                     = 'd''';
    F1.info.PF                     = @RPF_scaled_Weibull;
    F1.info.padCells               = 1;
    F1.info.set_P_max_to_d_pad_max = 1;

    F1.info.x_min                  = 0;
    F1.info.x_max                  = 1;
    F1.info.x_label                = 'coherence';

    F1.info.cond_labels            = {'low volatility', 'high volatility'};
    
    % RPF_update_info will infer nRatings from trialData.rating,
    % so you do not need to set F1.info.nRatings manually.

    % These padding options also have defaults in RPF_update_info.
    % If you do not see padCells/padAmount errors, you can ignore them.
    % F1.info.constrain.value.gamma = 0;
    % F1.info.constrain.value.omega = 'P_max';

    F1 = RPF_get_F(F1.info, trialData);

    %% 4. F2: P2(x) = p(high rating | coherence)  (confidence)

    F2 = struct();

    F2.info.DV          = 'p(high rating)';
    F2.info.DV_respCond = 'all';
    F2.info.PF          = @PAL_Weibull;

    F2.info.x_min       = 0;
    F2.info.x_max       = 1;
    F2.info.x_label     = 'coherence';
    F2.info.cond_labels = {'low volatility', 'high volatility'};
    F2.info.constrain   = [];

    F2 = RPF_get_F(F2.info, trialData);

    % Fit F2 (safe to call again)
    F2 = RPF_get_F(F2.info, trialData);

    %% 5. RPF: R(P1) = P2

    P1_LB = [];
    P1_UB = [];
    R     = RPF_get_R(F1, F2, P1_LB, P1_UB);

    %% 6. Plot RPF for this subject (standard RPF plot)

    plotSettings = struct();
    plotSettings.all.set_title_param = 1;
    plotSettings.F{1}.set_legend     = 1;
    plotSettings.str_sgtitle         = sprintf('Subject %d: performance-confidence RPF', thisSub);

    % You can choose which plot you like:
    % RPF_plot(R.F1, plotSettings, 'F');   % only F1 (performance PF)
    % RPF_plot(R.F2, plotSettings, 'F');   % only F2 (confidence PF)

    % Only RPF (performance → confidence)
    RPF_plot(R, plotSettings, 'R');

    % Or all panels:
    % RPF_plot(R, plotSettings, 'all');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now use RPF data points (R.F1.data / R.F2.data)
    % Draw lines ordered by coherence and label each coherence
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure; hold on;

    % Number of conditions (here 2: low / high volatility)
    nCond  = numel(R.F1.data);      % R.F1.data is 1x2 struct array
    colors = lines(nCond);

    for k = 1:nCond

        % For this condition, get coherence / d' / p(high rating)
        % All values come from RPF data (no extra fitting here)
        x_coh = R.F1.data(k).x(:);      % coherence list
        dvals = R.F1.data(k).P(:);      % d' values
        pvals = R.F2.data(k).P(:);      % p(high rating) values

        % Sort by coherence
        [coh_sorted, ord] = sort(x_coh);
        d_sorted = dvals(ord);
        p_sorted = pvals(ord);

        % Draw line with markers (just connecting the points)
        plot(d_sorted, p_sorted, '-o', ...
             'Color', colors(k,:), ...
             'LineWidth', 2, ...
             'MarkerFaceColor', 'w', ...
             'MarkerEdgeColor', colors(k,:));

        % Label each point with its coherence value
        for i = 1:numel(d_sorted)
            text(d_sorted(i) + 0.05, p_sorted(i), ...
                 sprintf('%.0f', coh_sorted(i)), ...
                 'Color', colors(k,:), 'FontSize', 8);
        end
    end

    xlabel('d'' (RPF data)');
    ylabel('p(high rating) (RPF data)');
    title(sprintf('Subject %d: RPF data with coh-ordered lines', thisSub));

    legend({R.F1.data.cond_label}, 'Location', 'southeast');
    grid on; box on;
    hold off;

end  % end of subject loop

% (Optional) Example of axis labels and legend if you only keep
% the coh-labeled perf–conf plot above:
xlabel('d'' (from RPF fit)');
ylabel('p(high rating) (from RPF fit)');
title(sprintf('Subject %d: RPF perf–conf with coh labels', thisSub));

legend(F1.info.cond_labels, 'Location', 'southeast');  % {'low volatility','high volatility'}
grid on; box on;
hold off;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 额外：为这个 subject 生成每个 trial 的预测正确率 p_perf_trial
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nTr = numel(coh_s);
    p_perf_trial = nan(nTr,1);   % 这个被试的每个 trial 的 p(correct)

    nCond = numel(R.F1.data);    % 这里应该是 2: low / high vol

    for c = 1:nCond
        % 当前 volatility 条件下的 trial
        mask_c = (condition_s == c);
        if ~any(mask_c)
            continue;
        end

        coh_c = coh_s(mask_c);          % 这些 trial 的 coherence
        % 对应 RPF F1 里这个条件的 performance 曲线
        x_grid = R.F1.data(c).x(:);     % 这一条件下的 coherences
        d_grid = R.F1.data(c).P(:);     % 对应的 d'(x)

        % --- 方法 A：coh_s 只取离散水平，直接匹配 ---
        % 如果 coh_s 和 x_grid 完全一样（比如都是 [0 .064 .128 ...]）
        % 可以用 ismember 找 index：
        [~, loc] = ismember(coh_c, x_grid);
        if any(loc == 0)
            warning('Some coherences not matched in F1.data; consider interp1.');
        end
        d_pred = d_grid(loc);

        % --- 方法 B（更稳）：如果担心有浮点误差，就用插值 ---
        % d_pred = interp1(x_grid, d_grid, coh_c, 'linear', 'extrap');

        % d' → p(correct)，这里假设是 2AFC
        p_corr = normcdf(d_pred ./ sqrt(2));

        % 填回到这个 subject 的 trial 向量里
        p_perf_trial(mask_c) = p_corr;
    end

    % 现在 p_perf_trial (nTr x 1) 就是这个被试每一个 trial 的预测正确率 p_j
    % 你可以保存到一个大的向量里，用于后面的 logistic 回归：

    if ~exist('p_perf_all', 'var')
        % 所有 valid trial 的总长度
        p_perf_all = nan(numel(valid), 1);
    end

    % 把这个被试的结果塞回原来 valid trial 的位置
    % 先找到 all trial 里属于这个被试、且 valid 的 index：
    idx_all_thisSub = find(valid);      % valid 的全局 index
    idx_all_thisSub = idx_all_thisSub(idxS);  % 只要这个被试的那些
    p_perf_all(idx_all_thisSub) = p_perf_trial;
