% ===== situation =====
cfg.CORR             = 'corr';        % 'all' / 'corr' / 'incorr'
cfg.DROP_HIGHEST_COH = true;         % remove coh == 512
cfg.REDEFINE_ANSWER = true;    % true = correct >= ME mean

% ===== predictor =====
cfg.P_PERF_MODE      = 'online';     % 'all' / 'online'
cfg.nBins            = 27;
cfg.winLen           = 5;
cfg.tol              = 1e-12;
cfg.useSubjDummies   = false;

% ===== model =====
cfg.MODEL_FAMILY     = 'blend';      % 'cond' / 'coh' / 'blend'
cfg.OUTCOME          = 'conf';       % 'conf' / 'acc' / 'rt'

% ===== quarter bar plot parameter =====
cfg.QUARTER_MODEL_MODE = 'manual';
cfg.QUARTER_MODEL_NAME = 'M4_twoWay_PxV';
cfg.QUARTER_TERM_NAME  = 'PxV';

% ===== output =====
cfg.outPDF_ab        = '../figure/AIC_BIC_bestModel_dots.pdf';
cfg.minN             = 50;
cfg.minN_sub         = 5;
cfg.sv_tol           = 1e-12;


% plot
cfg.DO_PLOT_AICBIC_DOTS = true;