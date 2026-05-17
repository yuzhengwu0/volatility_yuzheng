tiledlayout;
figure;

% plot z-scored confidence (outcome)
try
    nexttile;
    histogram(cfg.ConfY);
    title('OUTCOME: confidence z-scored within subject');
catch
end

% plot RT
try
    nexttile;
    histogram(cfg.rtX);
    title('z-logged RT (R predictor)')
catch
end

% plot correctness
try
    nexttile;
    histogram(cfg.Correct);
    title('accuracy (C predictor)');
catch
end

% plot coherence
try
    nexttile;
    histogram(cfg.coh_weuse);
    title('coherence (Coh predictor)')
catch
end

% plot performance term
try
    nexttile;
    switch cfg.P_PERF_MODE
        case 'online'
            histogram(cfg.z_perf);
            title('p perf online (P predictor)');
        case 'all'
            histogram(cfg.z_perf);
            title('p perf all (P predictor)');
    end
catch
end

% plot raw volatility
try
    nexttile;
    histogram(cfg.resVol_mat);
    title('raw volatility (not used)')
catch
end

% plot z-logged volatility
try
    nexttile;
    histogram(cfg.zlog_vol);
    title('zlogged volatility (V predictor)')
catch
end

% plot moco_diff (vol1)
try 
    nexttile;
    histogram(moco_diff);
    title('moco diff')
catch
end 

% plot vol2 signed
try
    nexttile;
    histogram(vol2_signed);
    title('vol2 signed')
catch 
end 

% plot vol2 abs
try
    nexttile;
    histogram(vol2_abs);
    title('vol2 abs')
catch 
end 
