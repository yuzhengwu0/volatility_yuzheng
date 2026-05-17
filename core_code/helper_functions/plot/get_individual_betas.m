%% get individual betas
coh_weuse = cfg.coh_weuse;
Correct = cfg. Correct;
cond = cfg.cond;
subjID = cfg.subjID;
ConfY = cfg.ConfY;

%fitted_models = struct();
y_var = ConfY; % 'confCont' / 'ConfY' / 'rtX'
x_var = coh_weuse;

acc_levels = unique(Correct);
cond_levels = unique(cond);
sub_levels = unique(subjID);

for a = 1:length(unique(Correct))
    acc = acc_levels(a);
    acc_idx = Correct == acc;
    for z = 1:length(unique(cond))
        zcond = cond_levels(z);
        volcond = cond == zcond;
        for s = 1:length(unique(subjID))
            subj = sub_levels(s);
            sub = subjID == subj;
            mask = acc_idx & volcond & sub;
            
            y = y_var(mask);
            x = x_var(mask);

            abc = fitlm(x, y);

            subject_betas(a, z, s) = abc.Coefficients.Estimate('x1');
            subject_SE(a, z, s) = abc.Coefficients.SE('x1');
        end

        group_mask = acc_idx & volcond;
        y = y_var(group_mask);
        x = x_var(group_mask);
        
        g = fitlm(x, y);
        group_betas(a, z) = g.Coefficients.Estimate('x1');
        group_SE(a, z) = g.Coefficients.SE('x1');
        
    end 
end 

%% plotting betas
colors    = {[0.5 0.8 0.5], [0.7 0.5 0.9]};   % green=incorrect, purple=correct
accLabels = {'Incorrect', 'Correct'};


figure;
hold on;
yline(0, 'HandleVisibility', 'off');

% change subject color
subjectColors = lines(size(subject_betas, 3));
for a = 1:2   % loop over accuracy
    color  = colors{a};

    % plot subject betas
    for s = 1:size(subject_betas, 3)
        scatter([1,2], squeeze(subject_betas(a,:,s)), ...
            40, subjectColors(s,:), 'filled', 'MarkerFaceAlpha', 0.3, 'HandleVisibility', 'off');
    end

    errorbar([1,2], group_betas(a,:), group_SE(a,:), 'o', ...
        'Color', color, 'MarkerFaceColor', color, ...
        'MarkerSize', 5, 'LineWidth', 2, 'CapSize', 6, ...
        'DisplayName', accLabels{a});

   plot([1,2], group_betas(a,:), '-', 'LineWidth', 2, 'Color', color, 'HandleVisibility', 'off')
end

xticks([1 2]);
xticklabels({'Low Volatility', 'High Volatility'});
xlim([0.5 2.5]);
ylabel('\beta_{coh}');
legend('Location', 'best');
title('Effect of coherence on z-scored confidence')

%% plot scatter
figure;
tiledlayout(2, 2);

for a = 1:length(acc_levels)
    acc = acc_levels(a);
    acc_idx = Correct == acc;

    for z = 1:length(cond_levels)
        zcond = cond_levels(z);
        volcond = cond == zcond;

        nexttile;
        hold on;

        for s = 1:length(sub_levels)
            subj = sub_levels(s);
            sub_mask = acc_idx & volcond & (subjID == subj);
            cohLevels = coh_weuse(sub_mask);
            subj_coh_means = arrayfun(@(c) mean(y_var(sub_mask & x_var == c)), cohLevels);
            scatter(cohLevels, subj_coh_means, 40, 'filled', 'MarkerFaceAlpha', 0.4);
        end

        title(['accuracy = ' num2str(acc) ', condition = ' num2str(zcond)]);
        xlabel('coherence');
        ylabel('mean confidence');
    end
end