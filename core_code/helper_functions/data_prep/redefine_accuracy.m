function cfg = redefine_accuracy(cfg)

    meanme = mean(cfg.evidence_strength, 2);
    for i = 1:length(cfg.req)
        if meanme(i) < 0
            if cfg.req(i) == min(cfg.req)
                cfg.req(i) = cfg.req(i) + 1;
            elseif cfg.req(i) == max(cfg.req)
                cfg.req(i) = cfg.req(i) - 1;
            end
        end
    end
    for i = 1:length(cfg.Correct)
        if cfg.req(i) == cfg.given(i)
            cfg.Correct(i) = 1;
        else
            cfg.Correct(i) = 0;
        end
    end
end