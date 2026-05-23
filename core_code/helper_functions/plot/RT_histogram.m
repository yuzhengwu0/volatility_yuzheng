%% RT histogram

subjID = cfg.subjID;
coh = cfg.coh;
cond = cfg.cond;
Correct = cfg.Correct;
rtX = cfg.rtX;


for s = 1:length(unique(subjID))

    unique_subj = sort(unique(subjID));
    this_subj = unique_subj(s);

    figure
    binEdges = linspace(min(rtX), max(rtX), 20);

    tiledlayout(length(unique(coh)), length(unique(Correct)));

    for c = 1:length(sort(unique(coh)))
        for corr = 1:length(unique(Correct))
            nexttile
            hold on
            % change ylim
            % ylim([0 14])
            xlabel("zlog RT")

            unique_coh = sort(unique(coh));
            this_coh = unique_coh(c);
            unique_corr = sort(unique(Correct));
            this_corr = unique_corr(corr);
            n = sum(subjID == this_subj & Correct == this_corr & ...
                coh == this_coh);

            title(sprintf('coh=%d, corr=%d, n=%d', this_coh, this_corr, n));
            for v = 1:length(unique(cond))

                unique_cond = sort(unique(cond));
                this_cond = unique_cond(v);

                idx = subjID == this_subj & Correct == this_corr & ...
                coh == this_coh & cond == this_cond;
                
                if v == min(cond)
                    histogram(rtX(idx), binEdges, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5)
                elseif v == max(cond)
                    histogram(rtX(idx), binEdges, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5)
                else
                    fprintf("ERROR!")
                end
            end
            hold off
        end
    end
    sgtitle(sprintf('zloged RT distribution, subj %d', this_subj));
end
