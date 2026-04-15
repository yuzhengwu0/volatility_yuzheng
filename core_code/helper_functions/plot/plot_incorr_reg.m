%% plot corr reg

% only use incorr
CORR = 'corr';
switch CORR
    case 'corr'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ~isnan(confCont_all) & correct_all == 1 & ...
            ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
    case 'incorr'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ~isnan(confCont_all) & correct_all == 0 & ...
            ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
    case 'all'
        valid_basic = ~isnan(coh_all) & ~isnan(correct_all) & ...
            ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
end

% ---- exclude coh == 5.12 -----
drop_highest_coh = true;
if drop_highest_coh
    valid_basic = ~isnan(coh_all) & coh_all ~= max(coh_all) & correct_all == 1 & ~isnan(correct_all) & ...
        ~isnan(confCont_all) & ~isnan(vol_all) & ~isnan(subjID_all) & ~isnan(rt_all) & allStruct.times_dots_on == 0.2;
end 

% ===== try low coh here =====
if DO_SPLIT_COH
    valid_coh = ismember(coh_all, LOW_COH_VALUES); %change here if we want to change to high coh trials
else
    valid_coh = true(size(coh_all));
end

valid = valid_basic & valid_coh;

coh           = coh_all(valid);
req           = req_all(valid);
given         = given_all(valid);
Correct       = correct_all(valid);
confCont      = confCont_all(valid);
vol           = vol_all(valid);
subjID        = subjID_all(valid);
motion_energy = motion_energy_all(valid);
rt            = rt_all(valid);

truesessiontrial = allStruct.trialnum(valid);
truesession = allStruct.session(valid);

nTrials = numel(coh);
fprintf('Total valid trials: %d\n', nTrials);

% ===================== 2. Prep predictors =====================
% prep z_score confidence

[ConfY, confCont] = transform_conf(confCont, subjID);

% prep z-score RT
if DO_USE_RT
    rtX = transform_rt(rt, subjID);
else
    rtX = [];

end

% PREP VOL
% convert to cond (1=low vol, 2=high vol)
vol_levels = unique(vol(~isnan(vol)));
if numel(vol_levels) ~= 2
    warning('Volatility levels are not 2. Check your data!');
end
cond = nan(size(vol));
cond(vol == min(vol_levels)) = 1;
cond(vol == max(vol_levels)) = 2;
%residual volatility & z-score
[resVol_mat, resVol, evidence_strength, volatility_strength] = compute_resVol(motion_energy, vol, nBins, winLen, tol, coh, cond, req);

% prep predicted performance
switch P_PERF_MODE
    case 'all'
        p_perf_all = compute_p_perf_all(subjID, cond, coh, Correct);
        z_perf = zscore(p_perf_all);
    case 'online'
        p_perf_online = compute_p_perf_online(subjID, cond, coh, Correct);
        z_perf = zscore(p_perf_online);
    case 'try' % if we want to see the distribution plot we can do all of them together, but usually do one at a time to avoid error
        p_perf_all = compute_p_perf_all(subjID, cond, coh, Correct);
        p_perf_online = compute_p_perf_online(subjID, cond, coh, Correct);
end

% accuracy --> we use correct directly

% coherence
coh_weuse = coh/100;
if DISCRETE_COH
    z_coh = coh_weuse >= median(coh_weuse);
else
    z_coh = zscore(coh_weuse);
end

% z-scored volatility condition
z_cond = zscore(cond);

% z-log momentary volatility
log_vol = log(resVol_mat + 1);
zlog_vol = zscore(log_vol);

% exclude coh == 5.12
% dropped_coh = 5.12;
% cond = cond(coh_weuse ~= dropped_coh);
% coh_weuse = coh_weuse(coh_weuse ~= dropped_coh);

%% do the dot plot
y_var = confCont; % we can change it to 'confCont', 'ConfY', 'rtX'
x_var = 1:length(unique(coh_weuse));
figure;

% coherence on x-axis, y-var on y-axis
% divide by high / low colatility

% two panel approach
figure;
for v = 1:2
    y_var = confCont(cond == v);
    x_var = coh_weuse(cond == v);
    nexttile;
    scatter(x_var, y_var);
end

%% one panel approach
conf_var = ConfY;
figure;
hold on
for v = 1:2
    % color points according to volatility
    if v == 1
        color = 'b';
    else
        color = 'r';
    end
    % get data just for volatility condition
    y_var = conf_var(cond == v);
    x_var = coh_weuse(cond == v);

    % scatter plot
    scatter(x_var, y_var, color);

    [p, S]        = polyfit(x_var, y_var, 1);
    [yFit, delta] = polyconf(p, x_var, S, 'alpha', 0.05);

    fill([x_var; flipud(x_var)], [yFit+delta; flipud(yFit-delta)], ...
        color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x_var, yFit, '-', 'Color', color, 'LineWidth', 2);

end
legend({'low volatility', 'high volatility'})
ylabel('raw confidence')
xlabel('coherence')

%% 
conf_var = rtX;
figure;
hold on
for v = 1:2
    % color points according to volatility
    if v == 1
        color = 'b';
    else
        color = 'r';
    end
    % get data just for volatility condition
    y_var = conf_var(cond == v);
    x_var = coh_weuse(cond == v);

    % scatter plot
    scatter(x_var, y_var, color);

    [x_sorted, idx] = sort(x_var);
    y_sorted        = y_var(idx);

    [p, S]        = polyfit(x_sorted, y_sorted, 1);
    [yFit, delta] = polyconf(p, x_sorted, S, 'alpha', 0.05);

    fill([x_sorted; flipud(x_sorted)], [yFit+delta; flipud(yFit-delta)], ...
        color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x_sorted, yFit, '-', 'Color', color, 'LineWidth', 2);

end

h1 = plot(nan, nan, 'b', 'LineWidth', 2);
h2 = plot(nan, nan, 'r', 'LineWidth', 2);
legend([h1 h2], {'low vol', 'high vol'})
ylabel('raw confidence')
xlabel('coherence')