%% correlation between RT and conf

% get relevant variables into the workspace
Correct = cfg.Correct;
cond = cfg.cond;
rtX = cfg.rtX;
ConfY = cfg.ConfY;
subj = cfg.subjID;
% get level-type variables for each var above
conditionType = unique(cond);
correctType = unique(Correct);
subjType = unique(subj);

% make table to store correlation coefficients & p-values
% index 1 (rows) = volatility condition (1=low, 2=high)
% index 2 (columns) = task type (1=RT, 2=deadline)
% index 3 (pages) = response type (1=all, 2=correct, 3=incorrect)
corr_coefs = NaN(2, 2, 3);
p_vals = corr_coefs;

% loop through condition combinations to make masks & compute corr

for vol = 1:length(conditionType)
    vol_mask = cond == conditionType(vol);
    for acc = 1:length(correctType)
        acc_mask = Correct == correctType(acc);
        for s = 1: length(unique(subj))
            subj_mask = subj == subjType(s);

            mask = acc_mask & vol_mask & subj_mask;
            rt = rtX(mask);
            conf = ConfY(mask);

            mean_conf(vol, acc, s) = mean(conf);
            mean_rt(vol, acc, s) = mean(rt);
            [corr_coefs(vol, acc, s), p_vals(vol, acc, s)] = corr(rt,conf);
        end
    end
end


% compute mean and standard error
SE = std(corr_coefs, 0, 3) / sqrt(size(corr_coefs, 3));

%% plot
% compute mean across subjects (dim 4)
mean_corr = mean(corr_coefs, 3); % 2x2x2

% labels
vol_labels = {'Low Vol', 'High Vol'};
acc_labels = {'Incorrect', 'Correct'};

% x-axis: all 2x2 combinations of vol x task
% build group labels and extract data
group_labels = {};
data_acc1 = [];
data_acc2 = [];
err_acc1  = [];
err_acc2  = [];


for vol = 1:2
    group_labels{end+1} = sprintf('%s\n%s', vol_labels{vol});
    data_acc1(end+1) = mean_corr(vol, 1);
    data_acc2(end+1) = mean_corr(vol, 2);
    err_acc1(end+1)  = SE(vol, 1);
    err_acc2(end+1)  = SE(vol, 2);
end


% plot
figure; hold on;

nGroups = 2;
x = 1:nGroups;
bar_width = 0.35;
offset = bar_width / 2;

b1 = bar(x - offset, data_acc1, bar_width, 'FaceColor', [0.7 0.3 0.7]);
b2 = bar(x + offset, data_acc2, bar_width, 'FaceColor', [0.3 0.6 0.3]);

% error bars
errorbar(x - offset, data_acc1, err_acc1, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
errorbar(x + offset, data_acc2, err_acc2, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);

% formatting
set(gca, 'XTick', x, 'FontSize', 12);
combined_labels = {'Low Vol', 'High Vol'};
xticklabels(combined_labels);
xtickangle(20);
ylabel('Correlation (RT vs Confidence)', 'FontSize', 13);
title('RT–Confidence Correlations by Task & Volatility', 'FontSize', 14);
legend(acc_labels, 'Location', 'best', 'FontSize', 11);
yline(0, '--k', 'Alpha', 0.4);
box off;
hold off;

%% plot with individual subject lines & points MEAN RT!!!
% compute mean across subjects (dim 4)
mean_corr = mean(mean_rt, 3); % 2x2x2

% labels
vol_labels = {'Low Vol', 'High Vol'};
task_labels = {'RT', 'Deadline'};
acc_labels = {'Incorrect', 'Correct'};
nSubj = size(mean_rt, 3);
subj_colors = lines(nSubj);

figure;
tl = tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'RT–Confidence Correlations by Task & Volatility', 'FontSize', 14);


    nexttile;
    hold on;

    % --- build vectors for this task ---
    grp_data_acc1 = squeeze(mean_corr(:, 1)); % [vol x 1]
    grp_data_acc2 = squeeze(mean_corr(:, 2));
    grp_err_acc1  = squeeze(SE(:, 1));
    grp_err_acc2  = squeeze(SE(:, 2));

    x = 1:2; % 2 vol conditions
    bar_width = 0.35;
    offset = bar_width / 2;

    % --- bars ---
    b1 = bar(x - offset, grp_data_acc1, bar_width, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5);
    b2 = bar(x + offset, grp_data_acc2, bar_width, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5);

    % --- group-level error bars ---
    errorbar(x - offset, grp_data_acc1, grp_err_acc1, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
    errorbar(x + offset, grp_data_acc2, grp_err_acc2, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);

    % --- individual subjects ---
    for s = 1:nSubj
        subj_acc1 = squeeze(mean_rt(:, 1, s)); % [vol x 1]
        subj_acc2 = squeeze(mean_rt(:, 2, s));

        for v = 1:2
            % connect correct to incorrect within each volatility condition
            plot([v - offset, v + offset], [subj_acc1(v), subj_acc2(v)], '-', ...
                'Color', [subj_colors(s,:) 0.6], 'LineWidth', 1.2);

            % correct dot
            plot(v - offset, subj_acc1(v), 'o', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);

            % incorrect dot
            plot(v + offset, subj_acc2(v), 's', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);
        end
    end

    % --- formatting ---
    set(gca, 'XTick', x, 'XTickLabel', {}, 'FontSize', 11);
    % for v = 1:2
    %     text(v, min(ylim) - 0.05 * range(ylim), ...
    %         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 11);
    % end

    yline(0, '--k', 'Alpha', 0.4);
    ylabel("mean RT")
    xlabel("low vol    high vol")
    ylim ([-0.3 0.7])
    % title(task_labels{t}, 'FontSize', 13);

    box off;
    hold off;


% --- shared legend ---
% grab handles from tile 1
nexttile(1);
h = get(gca, 'Children');

% build dummy handles on tile 1 for clean legend entries
hold on;
dummy(1) = bar(nan, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{1});
dummy(2) = bar(nan, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{2});
for s = 1:nSubj
    dummy(2+s) = plot(nan, nan, 'o-', ...
        'Color', subj_colors(s,:), ...
        'MarkerFaceColor', subj_colors(s,:), ...
        'DisplayName', sprintf('S%d', s));
end
hold off;

legend(dummy, 'Location', 'bestoutside', 'FontSize', 10);


%% plot with individual subject lines & points MEAN CONF!!!
% compute mean across subjects (dim 4)
mean_corr = mean(mean_conf, 3); % 2x2x2

% labels
vol_labels = {'Low Vol', 'High Vol'};
task_labels = {'RT', 'Deadline'};
acc_labels = {'Incorrect', 'Correct'};
nSubj = size(mean_conf, 3);
subj_colors = lines(nSubj);

figure;
tl = tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'RT–Confidence Correlations by Task & Volatility', 'FontSize', 14);


    nexttile;
    hold on;

    % --- build vectors for this task ---
    grp_data_acc1 = squeeze(mean_corr(:, 1)); % [vol x 1]
    grp_data_acc2 = squeeze(mean_corr(:, 2));
    grp_err_acc1  = squeeze(SE(:, 1));
    grp_err_acc2  = squeeze(SE(:, 2));

    x = 1:2; % 2 vol conditions
    bar_width = 0.35;
    offset = bar_width / 2;

    % --- bars ---
    b1 = bar(x - offset, grp_data_acc1, bar_width, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5);
    b2 = bar(x + offset, grp_data_acc2, bar_width, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5);

    % --- group-level error bars ---
    errorbar(x - offset, grp_data_acc1, grp_err_acc1, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
    errorbar(x + offset, grp_data_acc2, grp_err_acc2, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);

    % --- individual subjects ---
    for s = 1:nSubj
        subj_acc1 = squeeze(mean_conf(:, 1, s)); % [vol x 1]
        subj_acc2 = squeeze(mean_conf(:, 2, s));

        for v = 1:2
            % connect correct to incorrect within each volatility condition
            plot([v - offset, v + offset], [subj_acc1(v), subj_acc2(v)], '-', ...
                'Color', [subj_colors(s,:) 0.6], 'LineWidth', 1.2);

            % correct dot
            plot(v - offset, subj_acc1(v), 'o', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);

            % incorrect dot
            plot(v + offset, subj_acc2(v), 's', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);
        end
    end

    % --- formatting ---
    set(gca, 'XTick', x, 'XTickLabel', {}, 'FontSize', 11);
    % for v = 1:2
    %     text(v, min(ylim) - 0.05 * range(ylim), ...
    %         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 11);
    % end

    yline(0, '--k', 'Alpha', 0.4);
    ylabel("mean conf")
    xlabel("low vol    high vol")
    ylim ([-0.9 0.4])
    % title(task_labels{t}, 'FontSize', 13);

    box off;
    hold off;


% --- shared legend ---
% grab handles from tile 1
nexttile(1);
h = get(gca, 'Children');

% build dummy handles on tile 1 for clean legend entries
hold on;
dummy(1) = bar(nan, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{1});
dummy(2) = bar(nan, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{2});
for s = 1:nSubj
    dummy(2+s) = plot(nan, nan, 'o-', ...
        'Color', subj_colors(s,:), ...
        'MarkerFaceColor', subj_colors(s,:), ...
        'DisplayName', sprintf('S%d', s));
end
hold off;

legend(dummy, 'Location', 'bestoutside', 'FontSize', 10);


%% plot with individual subject lines & points
% compute mean across subjects (dim 4)
mean_corr = mean(corr_coefs, 3); % 2x2x2

% labels
vol_labels = {'Low Vol', 'High Vol'};
task_labels = {'RT', 'Deadline'};
acc_labels = {'Incorrect', 'Correct'};
nSubj = size(corr_coefs, 3);
subj_colors = lines(nSubj);

figure;
tl = tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'RT–Confidence Correlations by Task & Volatility', 'FontSize', 14);


    nexttile;
    hold on;

    % --- build vectors for this task ---
    grp_data_acc1 = squeeze(mean_corr(:, 1)); % [vol x 1]
    grp_data_acc2 = squeeze(mean_corr(:, 2));
    grp_err_acc1  = squeeze(SE(:, 1));
    grp_err_acc2  = squeeze(SE(:, 2));

    x = 1:2; % 2 vol conditions
    bar_width = 0.35;
    offset = bar_width / 2;

    % --- bars ---
    b1 = bar(x - offset, grp_data_acc1, bar_width, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5);
    b2 = bar(x + offset, grp_data_acc2, bar_width, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5);

    % --- group-level error bars ---
    errorbar(x - offset, grp_data_acc1, grp_err_acc1, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
    errorbar(x + offset, grp_data_acc2, grp_err_acc2, 'k', ...
        'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);

    % --- individual subjects ---
    for s = 1:nSubj
        subj_acc1 = squeeze(corr_coefs(:, 1, s)); % [vol x 1]
        subj_acc2 = squeeze(corr_coefs(:, 2, s));

        for v = 1:2
            % connect correct to incorrect within each volatility condition
            plot([v - offset, v + offset], [subj_acc1(v), subj_acc2(v)], '-', ...
                'Color', [subj_colors(s,:) 0.6], 'LineWidth', 1.2);

            % correct dot
            plot(v - offset, subj_acc1(v), 'o', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);

            % incorrect dot
            plot(v + offset, subj_acc2(v), 's', ...
                'Color', subj_colors(s,:), 'MarkerFaceColor', subj_colors(s,:), 'MarkerSize', 6);
        end
    end

    % --- formatting ---
    set(gca, 'XTick', x, 'XTickLabel', {}, 'FontSize', 11);
    % for v = 1:2
    %     text(v, min(ylim) - 0.05 * range(ylim), vol_labels{v}, ...
    %         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 11);
    % end

    yline(0, '--k', 'Alpha', 0.4);
    ylabel("RT-conf corr coef")
    xlabel("low vol     high vol")
    ylim([-0.7 0])
    % title(task_labels{t}, 'FontSize', 13);

    box off;
    hold off;


% --- shared legend ---
% grab handles from tile 1
nexttile(1);
h = get(gca, 'Children');

% build dummy handles on tile 1 for clean legend entries
hold on;
dummy(1) = bar(nan, 'FaceColor', [0.3 0.6 0.3], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{1});
dummy(2) = bar(nan, 'FaceColor', [0.7 0.3 0.7], 'FaceAlpha', 0.5, 'DisplayName', acc_labels{2});
for s = 1:nSubj
    dummy(2+s) = plot(nan, nan, 'o-', ...
        'Color', subj_colors(s,:), ...
        'MarkerFaceColor', subj_colors(s,:), ...
        'DisplayName', sprintf('S%d', s));
end
hold off;

legend(dummy, 'Location', 'bestoutside', 'FontSize', 10);
%% all
figure
for v = 1:length(unique(cond))
    for t = 1:2
        nexttile
        hold on
        x = cond(v);
        if t == 1
            this_rt = rtX(valid_fixed) == 0;

            y = t;
            if correct == 1
                z = 1;
            else
                for c = 1:length(unique(Correct))
                    z = c + 1;
                end
            end

            this_rt = rtX(x) & ;

            % NEED TO ADD MASK BEFORE COMPUTING CORR
            [corr_coefs(x, y, z), p_vals(x, y, z)] = corr(rtX,ConfY);
        end
    end

    % for corr = 1:length(unique(Correct))

