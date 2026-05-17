
data_path = 'fit_rs_meta_d_MLE.m/';

coh = cfg.coh;
cond = cfg.cond;
subjID = cfg.subjID;
cohlevel = unique(coh);
condlevel = unique(cond);
subjIDlevel = unique(subjID);


conflevel = zeros(length(confCont), 1);
conflevel(confCont <= 0.25)                            = 1;
conflevel(confCont > 0.25 & confCont <= 0.50)          = 2;
conflevel(confCont > 0.50 & confCont <= 0.75)          = 3;
conflevel(confCont > 0.75)                             = 4;

for s = 1:length(subjIDlevel)
    for v = 1:length(condlevel)      % loop over unique conditions
        for c = 1:length(cohlevel)   % loop over unique coh levels
    
            idx = (subjID == subjIDlevel(s)) & (cond == condlevel(v)) & (coh == cohlevel(c));
    
            req_grp   = req(idx);
            given_grp = given(idx);
            conf_grp  = conflevel(idx);
    
            fprintf('Cond %d | Coh %.0f | Trials: %d\n', ...
                condlevel(v), cohlevel(c), sum(idx));
    
            counts = accumarray([req_grp, given_grp, conf_grp], 1, [2, 2, 4]);
    
            nR_S1 = [counts(1,1,4) counts(1,1,3) counts(1,1,2) counts(1,1,1) ...
                     counts(1,2,1) counts(1,2,2) counts(1,2,3) counts(1,2,4)];
    
            % build nR_S2: [miss high→low | hit low→high]
            nR_S2 = [counts(2,1,4) counts(2,1,3) counts(2,1,2) counts(2,1,1) ...
                     counts(2,2,1) counts(2,2,2) counts(2,2,3) counts(2,2,4)];
    
            fprintf('Cond %d | Coh %.2f | Trials: %d\n', condlevel(v), cohlevel(c), sum(idx));
            fprintf('  nR_S1: %s\n', num2str(nR_S1));
            fprintf('  nR_S2: %s\n\n', num2str(nR_S2));
    
    
            % --- zero cell correction ---
            if any(nR_S1 == 0) || any(nR_S2 == 0)
                adj = 1 / length(nR_S1);   % = 0.125
                nR_S1 = nR_S1 + adj;
                nR_S2 = nR_S2 + adj;
            end
    
            % --- fit meta-d' ---
            try
                fit = fit_rs_meta_d_MLE(nR_S1, nR_S2);
    
                % extract key measures
                da      = fit.da;
                meta_da = (fit.meta_da_rS1 + fit.meta_da_rS2) / 2;
                M_ratio = meta_da / da;
    
                % store results
                results(s, v, c).subjIDlevel = subjIDlevel(s);
                results(s, v, c).condlevel = condlevel(v);
                results(s, v, c).cohlevel  = cohlevel(c);
                results(s, v, c).da        = da;
                results(s, v, c).meta_da   = meta_da;
                results(s, v, c).M_ratio   = M_ratio;
    
                fprintf('Subj %d | Cond %d | Coh %.0f | da=%.3f | meta-da=%.3f | M-ratio=%.3f\n', ...
                    subjIDlevel(s), condlevel(v), cohlevel(c), da, meta_da, M_ratio);
            catch e
                fprintf('Subj %d | Cond %d | Coh %.0f | FIT FAILED: %s\n', ...
                    subjIDlevel(s), condlevel(v), cohlevel(c), e.message);
    

                results(s, v, c).subjIDlevel = subjIDlevel(s);
                results(s, v, c).condlevel = condlevel(v);
                results(s, v, c).cohlevel  = cohlevel(c);
                results(s, v, c).da        = NaN;
                results(s, v, c).meta_da   = NaN;
                results(s, v, c).M_ratio   = NaN;
            end
        end 
    end
end



%% plot - m-ratio
figure; hold on;
colors = {[0.2 0.4 0.8], [0.8 0.2 0.2]};

for v = 1:length(condlevel)
    m_ratios = arrayfun(@(c) results(v,c).M_ratio, 1:length(cohlevel));
    plot(1:length(cohlevel), m_ratios, '-o', ...
         'Color', colors{v}, 'MarkerFaceColor', colors{v}, ...
         'LineWidth', 2, 'MarkerSize', 8, ...
         'DisplayName', sprintf('Cond %d', condlevel(v)));
end

yline(1, '--k', 'Label', 'Ideal (M=1)');
xticks(1:length(cohlevel));
xticklabels(arrayfun(@num2str, cohlevel, 'UniformOutput', false));
xlabel('Coherence Level'); ylabel('M-ratio (meta-d'' / d'')');
title('M-ratio as a Function of Coherence');
legend; grid on; box on; ylim([0 3]);





%% plot - with individual subjects + mean + SE
figure;
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
condColors  = {[0.2 0.4 0.8], [0.8 0.2 0.2]};   % blue, red
condLabels  = {'Low Volatility', 'High Volatility'};
xvals       = 1:length(cohlevel);
nCoh        = length(cohlevel);
nCond       = length(condlevel);
nSubj       = length(subjIDlevel);
jitter_amt  = 0.08;   % horizontal spread for individual dots

% --- extract per-subject data from results_subj(subj, cond, coh) ---
% assumes you have results_subj(s, v, c).da / .meta_da / .M_ratio
% if your struct is named differently, change here

da_all      = NaN(nSubj, nCond, nCoh);
meta_all    = NaN(nSubj, nCond, nCoh);
mratio_all  = NaN(nSubj, nCond, nCoh);

for s = 1:nSubj
    for v = 1:nCond
        for c = 1:nCoh
            da_all(s,v,c)     = results(s,v,c).da;
            meta_all(s,v,c)   = results(s,v,c).meta_da;
            mratio_all(s,v,c) = results(s,v,c).M_ratio;
        end
    end
end

metrics     = {da_all, meta_all, mratio_all};
ylabels     = {"d'", "meta-d'", "M-ratio"};
titles      = {"d' (Perceptual Sensitivity)", ...
               "meta-d' (Metacognitive Sensitivity)", ...
               "M-ratio (Metacognitive Efficiency)"};
ylims       = {[-1 6], [-1 6], [-1 6]};

for m = 1:3
    nexttile;
    hold on;
    data = metrics{m};   % nSubj x nCond x nCoh

    for v = 1:nCond
        baseColor  = condColors{v};
        lightColor = baseColor * 0.4 + 0.6;  % lighter version for individual dots

        % compute mean and SE across subjects
        vals      = squeeze(data(:, v, :));   % nSubj x nCoh
        mu        = mean(vals, 1, 'omitnan'); % 1 x nCoh
        se        = std(vals, 0, 1, 'omitnan') / sqrt(nSubj);

        % individual subject dots (light color, jittered)
        for s = 1:nSubj
            jitter = (s - (nSubj+1)/2) * jitter_amt;
            plot(xvals + jitter, squeeze(data(s, v, :))', 'o', ...
                'Color',           lightColor, ...
                'MarkerFaceColor', lightColor, ...
                'MarkerSize',      3, ...
                'HandleVisibility','off');
        end

        % mean line + solid dot
        plot(xvals, mu, '-o', ...
            'Color',           baseColor, ...
            'MarkerFaceColor', baseColor, ...
            'LineWidth',       1.5, ...
            'MarkerSize',      1.5, ...
            'DisplayName',     condLabels{v});

        % SE error bars
        errorbar(xvals, mu, se, ...
            'Color',            baseColor, ...
            'LineStyle',        'none', ...
            'LineWidth',        2, ...
            'CapSize',         3, ...
            'HandleVisibility', 'off');
    end

    % M-ratio reference line
    if m == 3
        yline(1, '--k', 'Alpha', 0.4, 'Label', 'Ideal (M=1)', ...
              'HandleVisibility', 'off');
    end

    ylabel(ylabels{m});
    title(titles{m});
    ylim(ylims{m});
    grid on; box on;
    legend('Location', 'best');
    xticks(xvals);
    if m < 3
        xticklabels([]);
    else
        xticklabels(arrayfun(@num2str, cohlevel, 'UniformOutput', false));
        xtickangle(30);
        xlabel('Coherence Level');
    end
end

sgtitle('Metacognitive Measures as a Function of Coherence', ...
        'FontSize', 14, 'FontWeight', 'bold');
set(gcf, 'Position', [100 100 700 800]);









