%% motion energy mean/STD/resvol plot
%evidence_strength = cfg.evidence_strength;
%volatility_strength = cfg.volatility_strength;
%resVol = cfg.resVol;
coh = cfg.coh;
cond = cfg.cond;
motion_mat = cfg.motion_mat;
motion_energy = cfg.motion_energy;


%% raw motion energy plot - by vol and coh



%% single trial (two trials)
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

clear yline
figure;
tiledlayout(length(unique(coh)), 2)

thiscoh = unique(coh);
for icoh = 1:numel(thiscoh)

    % raw motion energy
    nexttile;
    hold on;
    % cond == 1
    idx1 = (coh == thiscoh(icoh)) & (cond == 1);
    data1 = motion_mat(idx1, :);
    plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    mu1 = mean(data1, 1, 'omitnan');
    % sd1 = std(data1, 0, 1, 'omitnan');

    % cond == 2
    idx2 = (coh == thiscoh(icoh)) & (cond == 2);
    data2 = motion_mat(idx2, :);
    plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    mu2 = mean(data2, 1, 'omitnan');
    % sd2 = std(data2, 0, 1, 'omitnan');

    x = 1:size(motion_mat, 2);
    plot(x, mu1, 'b', 'LineWidth', 1.5)
    plot(x, mu2, 'r', 'LineWidth', 1.5)
    % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);

    ylim([-0.0005 0.0005])
    title(sprintf('coh = %g', thiscoh(icoh)))
    xlabel('windows')
    ylabel('raw motion energy')
    yline(0, 'HandleVisibility', 'off');




    %evidence strength
    nexttile;
    hold on;
    % cond == 1
    idx1 = (coh == thiscoh(icoh)) & (cond == 1);
    data1 = motion_diff(idx1, :);
    plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    mu1 = mean(data1, 1, 'omitnan');
    % sd1 = std(data1, 0, 1, 'omitnan');

    % cond == 2
    idx2 = (coh == thiscoh(icoh)) & (cond == 2);
    data2 = motion_diff(idx2, :);
    plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    mu2 = mean(data2, 1, 'omitnan');
    % sd2 = std(data2, 0, 1, 'omitnan');

    x = 1:size(motion_diff, 2);
    plot(x, mu1, 'b', 'LineWidth', 1.5)
    plot(x, mu2, 'r', 'LineWidth', 1.5)
    % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);

    ylim([-0.0002 0.0005])
    title(sprintf('coh = %g', thiscoh(icoh)))
    xlabel('windows')
    ylabel('motion diff')
    yline(0, 'HandleVisibility', 'off');

    % %volatility strength
    % nexttile;
    % hold on;
    % % cond == 1
    % idx1 = (coh == thiscoh(icoh)) & (cond == 1);
    % data1 = volatility_strength(idx1, :);
    % plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    % mu1 = mean(data1, 1, 'omitnan');
    % % sd1 = std(data1, 0, 1, 'omitnan');
    % 
    % % cond == 2
    % idx2 = (coh == thiscoh(icoh)) & (cond == 2);
    % data2 = volatility_strength(idx2, :);
    % plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    % mu2 = mean(data2, 1, 'omitnan');
    % % sd2 = std(data2, 0, 1, 'omitnan');
    % 
    % x = 1:size(volatility_strength, 2);
    % plot(x, mu1, 'b', 'LineWidth', 1.5)
    % plot(x, mu2, 'r', 'LineWidth', 1.5);
    % % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);
    % 
    % ylim([0 0.0002])
    % title(sprintf('coh = %g', thiscoh(icoh)))
    % xlabel('windows')
    % ylabel('volatility strength')
    % yline(0, 'HandleVisibility', 'off');
    % 
    % % resVol
    % nexttile;
    % hold on;
    % % cond == 1
    % idx1 = (coh == thiscoh(icoh)) & (cond == 1);
    % data1 = resVol(idx1, :);
    % plot(data1', 'Color', [0 0 1 0.05], 'LineWidth', 0.005);
    % mu1 = mean(data1, 1, 'omitnan');
    % % sd1 = std(data1, 0, 1, 'omitnan');
    % 
    % % cond == 2
    % idx2 = (coh == thiscoh(icoh)) & (cond == 2);
    % data2 = resVol(idx2, :);
    % plot(data2', 'Color', [1 0 0 0.05], 'LineWidth', 0.005);
    % mu2 = mean(data2, 1, 'omitnan');
    % % sd2 = std(data2, 0, 1, 'omitnan');
    % 
    % x = 1:size(resVol, 2);
    % plot(x, mu1, 'b', 'LineWidth', 1.5)
    % plot(x, mu2, 'r', 'LineWidth', 1.5)
    % % errorbar(x, mu1, sd1, 'b', 'LineWidth', 1.5);
    % % errorbar(x, mu2, sd2, 'r', 'LineWidth', 1.5);
    % 
    % ylim([-2.5 5])
    % title(sprintf('coh = %g', thiscoh(icoh)))
    % xlabel('windows')
    % ylabel('resVol')
    % yline(0, 'HandleVisibility', 'off');
end
h1 = plot(nan, nan, 'b', 'LineWidth', 2);
h2 = plot(nan, nan, 'r', 'LineWidth', 2);
legend([h1 h2], {'low vol', 'high vol'})



%% trial by trial coherence panel plot (gen by llm)
% 
% figure;
% tiledlayout('flow');
% 
% thiscoh = unique(coh);
% 
% for tr = 1:size(resVol, 1)
% 
%     % which coherence level
%     icoh = find(thiscoh == coh(tr));
%     nexttile(icoh);
%     hold on;
% 
%     % different color for different vol condition
%     if cond(tr) == 1
%         this_col = [0 0 1];   % blue
%     elseif cond(tr) == 2
%         this_col = [1 0 0];   % red
%     else
%         continue
%     end
% 
%     plot(resVol(tr, :), '-', 'Color', this_col, 'LineWidth', 0.8);
% 
%     yline(0, 'HandleVisibility', 'off');
%     ylim([-2.5 5])
%     title(sprintf('coh = %g', coh(tr)))
%     xlabel('windows')
%     ylabel('resVol')
% 
%     sgtitle(sprintf('Current trial = %d | blue = cond 1 | red = cond 2', tr))
%     drawnow;
%     pause;
% end