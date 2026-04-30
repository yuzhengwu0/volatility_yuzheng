% -------------------------------------------------------------------------
subjID = cfg.subjID;
z_cond = cfg.cond;
Correct = cfg.Correct;
coh_weuse = cfg.coh_weuse;
confCont = cfg.confCont;



% FLAG: set to 'volatility', 'accuracy', or 'both'
panel_by = 'accuracy';
% change y-axis variable
y_var = confCont; % we can change it to 'confCont', 'ConfY', 'rtX'
% -------------------------------------------------------------------------

% put all data into a table for easy grouping
allData = table(subjID, y_var, Correct, coh_weuse, z_cond);
% make subject categorical variable
allData = convertvars(allData, ["subjID", "Correct", "z_cond", "coh_weuse"], "categorical");

% Rename z_cond to volatility labels early
zNumeric_all = str2double(string(allData.z_cond));
volatilityLabel_all = repmat("", height(allData), 1);
volatilityLabel_all(zNumeric_all == 1) = "Low Volatility";
volatilityLabel_all(zNumeric_all == 2) = "High Volatility";
allData.volatility = categorical(cellstr(volatilityLabel_all), {'Low Volatility', 'High Volatility'});

% mean confidence in each accuracy x coherence x condition cell
subjectMeans = groupsummary(allData, ["Correct", "coh_weuse", "volatility", "subjID"], "mean", "y_var");
groupMeans = groupsummary(subjectMeans, ["Correct", "coh_weuse", "volatility"], ["mean", "std"], "mean_y_var");

% Back-transform coh_weuse from categorical to numeric
cohNumeric     = str2double(string(groupMeans.coh_weuse));
cohNumeric_sub = str2double(string(subjectMeans.coh_weuse));

% Define configs for each row
configs = {};
if strcmp(panel_by, 'volatility') || strcmp(panel_by, 'both')
    configs{end+1} = struct(...
        'panelLevels',  {{'Low Volatility', 'High Volatility'}}, ...
        'panelTitles',  {{'Low Volatility', 'High Volatility'}}, ...
        'colorLevels',  {unique(groupMeans.Correct)}, ...
        'colorLabels',  {{'Incorrect', 'Correct'}}, ...
        'panelVar',     groupMeans.volatility, ...
        'panelVar_sub', subjectMeans.volatility, ...
        'colorVar',     groupMeans.Correct, ...
        'colorVar_sub', subjectMeans.Correct, ...
        'colors',       {{[0.3 0.6 0.3], [0.7 0.3 0.7]}});  % green=incorrect, purple=correct
end
if strcmp(panel_by, 'accuracy') || strcmp(panel_by, 'both')
    configs{end+1} = struct(...
        'panelLevels',  {unique(groupMeans.Correct)}, ...
        'panelTitles',  {{'Incorrect', 'Correct'}}, ...
        'colorLevels',  {{'Low Volatility', 'High Volatility'}}, ...
        'colorLabels',  {{'Low Volatility', 'High Volatility'}}, ...
        'panelVar',     groupMeans.Correct, ...
        'panelVar_sub', subjectMeans.Correct, ...
        'colorVar',     groupMeans.volatility, ...
        'colorVar_sub', subjectMeans.volatility, ...
        'colors',       {{[0.2 0.4 0.8], [0.8 0.2 0.2]}});  % blue=low, red=high
end

cohLevels = sort(unique(cohNumeric));
nCoh      = numel(cohLevels);
barWidth  = 0.35;
offsets   = [-0.5, 0.5] * barWidth;
nRows     = numel(configs);
nCols     = 2;

figure;
tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

axs = gobjects(nRows, nCols);

for r = 1:nRows
    plotcfg = configs{r};

    for f = 1:numel(plotcfg.panelLevels)
        axs(r, f) = nexttile;
        hold on;

        for k = 1:numel(plotcfg.colorLevels)
            yVals   = nan(nCoh, 1);
            semVals = nan(nCoh, 1);
            
            % collect subject data for this color level across all coherences
            ySubj_all  = cell(nCoh, 1);
            xJitter_all = cell(nCoh, 1);

            for c = 1:nCoh
                % Group mean + SEM
                mask = plotcfg.panelVar == plotcfg.panelLevels(f) & ...
                       cohNumeric == cohLevels(c) & ...
                       plotcfg.colorVar == plotcfg.colorLevels(k);

                if any(mask)
                    yVals(c)   = groupMeans.mean_mean_y_var(mask);
                    n          = groupMeans.GroupCount(mask);
                    semVals(c) = groupMeans.std_mean_y_var(mask) / sqrt(n);
                end

                % Collect subject points for later
                subMask = plotcfg.panelVar_sub == plotcfg.panelLevels(f) & ...
                          cohNumeric_sub == cohLevels(c) & ...
                          plotcfg.colorVar_sub == plotcfg.colorLevels(k);

                if any(subMask)
                    ySubj_all{c} = subjectMeans.mean_y_var(subMask);
                    xJitter_all{c} = (c + offsets(k)) + ...
                                     (rand(numel(ySubj_all{c}), 1) - 0.5) * barWidth * 0.4;
                end
            end

            % Bars
            xPos = (1:nCoh) + offsets(k);
            bar(xPos, yVals, barWidth, ...
                'FaceColor', plotcfg.colors{k}, ...
                'FaceAlpha', 0.7, ...
                'DisplayName', plotcfg.colorLabels{k});

            % Error bars
            errorbar(xPos, yVals, semVals, ...
                'k', 'LineStyle', 'none', ...
                'LineWidth', 1.5, ...
                'CapSize', 6, ...
                'HandleVisibility', 'off');

            % Subject points on top
            for c = 1:nCoh
                if ~isempty(ySubj_all{c})
                    scatter(xJitter_all{c}, ySubj_all{c}, 30, plotcfg.colors{k}, ...
                        'filled', 'MarkerFaceAlpha', 0.4, ...
                        'MarkerEdgeAlpha', 0, ...
                        'HandleVisibility', 'off');
                end
            end
        end

        hold off;
        xlabel('Coherence Level');
        ylabel('Mean Confidence');
        title(plotcfg.panelTitles{f});
        xticks(1:nCoh);
        xticklabels(string(cohLevels));
        legend('Location', 'best');
        box on;
    end

    % Link y-axes within each row
    linkaxes(axs(r, :), 'y');
end

%% -------------------------------------------------------------------------
% Figure 2: highest coherence level only
% -------------------------------------------------------------------------
% maxCoh    = max(cohLevels);
% maxCohIdx = find(cohLevels == maxCoh);
% 
% figure;
% tiledlayout(nRows, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
% axs2 = gobjects(nRows, 1);
% 
% for r = 1:nRows
%     plotcfg = configs{r};
%     axs2(r) = nexttile;
%     hold on;
% 
%     for k = 1:numel(plotcfg.colorLevels)
% 
%         % Group mean + SEM
%         mask = plotcfg.panelVar == plotcfg.panelLevels(1) & ...  % dummy - overwritten below
%                cohNumeric == maxCoh & ...
%                plotcfg.colorVar == plotcfg.colorLevels(k);
% 
%         % collect across panels (i.e. across volatility or accuracy levels)
%         yVals   = nan(numel(plotcfg.panelLevels), 1);
%         semVals = nan(numel(plotcfg.panelLevels), 1);
%         ySubj_all   = cell(numel(plotcfg.panelLevels), 1);
%         xJitter_all = cell(numel(plotcfg.panelLevels), 1);
% 
%         for f = 1:numel(plotcfg.panelLevels)
%             mask = plotcfg.panelVar == plotcfg.panelLevels(f) & ...
%                    cohNumeric == maxCoh & ...
%                    plotcfg.colorVar == plotcfg.colorLevels(k);
% 
%             if any(mask)
%                 yVals(f)   = groupMeans.mean_mean_y_var(mask);
%                 n          = groupMeans.GroupCount(mask);
%                 semVals(f) = groupMeans.std_mean_y_var(mask) / sqrt(n);
%             end
% 
%             subMask = plotcfg.panelVar_sub == plotcfg.panelLevels(f) & ...
%                       cohNumeric_sub == maxCoh & ...
%                       plotcfg.colorVar_sub == plotcfg.colorLevels(k);
% 
%             if any(subMask)
%                 ySubj_all{f}   = subjectMeans.mean_y_var(subMask);
%                 xJitter_all{f} = (f + offsets(k)) + ...
%                                  (rand(numel(ySubj_all{f}), 1) - 0.5) * barWidth * 0.4;
%             end
%         end
% 
%         % Bars
%         xPos = (1:numel(plotcfg.panelLevels)) + offsets(k);
%         bar(xPos, yVals, barWidth, ...
%             'FaceColor', plotcfg.colors{k}, ...
%             'FaceAlpha', 0.7, ...
%             'DisplayName', plotcfg.colorLabels{k});
% 
%         % Error bars
%         errorbar(xPos, yVals, semVals, ...
%             'k', 'LineStyle', 'none', ...
%             'LineWidth', 1.5, ...
%             'CapSize', 6, ...
%             'HandleVisibility', 'off');
% 
%         % Subject points on top
%         for f = 1:numel(plotcfg.panelLevels)
%             if ~isempty(ySubj_all{f})
%                 scatter(xJitter_all{f}, ySubj_all{f}, 30, plotcfg.colors{k}, ...
%                     'filled', 'MarkerFaceAlpha', 0.4, ...
%                     'MarkerEdgeAlpha', 0, ...
%                     'HandleVisibility', 'off');
%             end
%         end
%     end
% 
%     hold off;
%     xlabel('');
%     ylabel('Mean Confidence');
%     title(sprintf('Highest Coherence (%.2f) — grouped by %s', maxCoh, ...
%           plotcfg.panelTitles{1}(1:find(isspace(plotcfg.panelTitles{1}),1)-1)));
%     xticks(1:numel(plotcfg.panelLevels));
%     xticklabels(plotcfg.panelTitles);
%     legend('Location', 'best');
%     box on;
% end
% 
% linkaxes(axs2, 'y');


%% -------------------------------------------------------------------------
% Mean Accuracy by Coherence x Volatility (single panel)
% -------------------------------------------------------------------------

% Put all data into a table
allData = table(subjID, Correct, coh_weuse, z_cond);
allData = convertvars(allData, ["subjID", "Correct", "z_cond", "coh_weuse"], "categorical");

% Rename z_cond to volatility labels
zNumeric_all = str2double(string(allData.z_cond));
volatilityLabel_all = repmat("", height(allData), 1);
volatilityLabel_all(zNumeric_all == 1) = "Low Volatility";
volatilityLabel_all(zNumeric_all == 2) = "High Volatility";
allData.volatility = categorical(cellstr(volatilityLabel_all), {'Low Volatility', 'High Volatility'});

% Convert Correct to numeric (0/1) for mean accuracy
allData.correctNum = double(allData.Correct == categorical({'1'}));

% Subject-level means, then group means
subjectMeans = groupsummary(allData, ["coh_weuse", "volatility", "subjID"], "mean", "correctNum");
groupMeans   = groupsummary(subjectMeans, ["coh_weuse", "volatility"], ["mean", "std"], "mean_correctNum");

% Back-transform coh_weuse to numeric for sorting/plotting
cohNumeric     = str2double(string(groupMeans.coh_weuse));
cohNumeric_sub = str2double(string(subjectMeans.coh_weuse));

cohLevels = sort(unique(cohNumeric));
nCoh      = numel(cohLevels);

volLevels = {'Low Volatility', 'High Volatility'};
colors    = {[0.2 0.4 0.8], [0.8 0.2 0.2]};   % blue = low, red = high
barWidth  = 0.35;
offsets   = [-0.5, 0.5] * barWidth;

figure;
hold on;

for k = 1:2
    yVals   = nan(nCoh, 1);
    semVals = nan(nCoh, 1);
    ySubj_all   = cell(nCoh, 1);
    xJitter_all = cell(nCoh, 1);

    for c = 1:nCoh
        % Group mean + SEM
        mask = string(groupMeans.volatility) == volLevels{k} & ...
               cohNumeric == cohLevels(c);
        if any(mask)
            yVals(c)   = groupMeans.mean_mean_correctNum(mask);
            n          = groupMeans.GroupCount(mask);
            semVals(c) = groupMeans.std_mean_correctNum(mask) / sqrt(n);
        end

        % Subject-level points
        subMask = string(subjectMeans.volatility) == volLevels{k} & ...
                  cohNumeric_sub == cohLevels(c);
        if any(subMask)
            ySubj_all{c}   = subjectMeans.mean_correctNum(subMask);
            xJitter_all{c} = (c + offsets(k)) + ...
                             (rand(numel(ySubj_all{c}), 1) - 0.5) * barWidth * 0.4;
        end
    end

    xPos = (1:nCoh) + offsets(k);

    % Bars
    bar(xPos, yVals, barWidth, ...
        'FaceColor', colors{k}, ...
        'FaceAlpha', 0.7, ...
        'DisplayName', volLevels{k});

    % Error bars
    errorbar(xPos, yVals, semVals, ...
        'k', 'LineStyle', 'none', ...
        'LineWidth', 1.5, ...
        'CapSize', 6, ...
        'HandleVisibility', 'off');

    % Subject-level scatter
    for c = 1:nCoh
        if ~isempty(ySubj_all{c})
            scatter(xJitter_all{c}, ySubj_all{c}, 30, colors{k}, ...
                'filled', 'MarkerFaceAlpha', 0.4, ...
                'MarkerEdgeAlpha', 0, ...
                'HandleVisibility', 'off');
        end
    end
end

hold off;
xlabel('Coherence Level');
ylabel('Mean Accuracy');
title('Mean Accuracy by Coherence and Volatility');
xticks(1:nCoh);
xticklabels(string(cohLevels));
ylim([0 1]);
legend('Location', 'best');
box on;