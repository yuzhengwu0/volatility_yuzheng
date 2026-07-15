% ===== situation =====
cfg.CORR             = 'all';        % 'all' / 'corr' / 'incorr'
% cfg.COHHH            = '512';      % '0', '32', '64', '128', '256', '512' / 'all'
cfg.DROP_HIGHEST_COH = false;         % remove coh == 512
cfg.REDEFINE_ANSWER  = false;    % true = correct >= ME mean
cfg.FLIPPING         = true;
cfg.HIGHVOL          = false;
cfg.VOLMODE          = 'abcde';   % 'vol1' - difference from trial_coh; 'vol2' - difference of last frame
                                 % 'vol3dir' - cumulative with direction;
                                 % 'vol3abs' - cumulative with abs val
                                 % 'abcde' - motion energy difference
VOL_USE_ME           = true;
cfg.RTtask           = false;
cfg.ALL        = false;

% ===== predictor =====
cfg.nBins            = 35;
if cfg.RTtask == false
    cfg.nBins            = 38;
end 
cfg.winLen           = 5;
cfg.tol              = 1e-12;
cfg.useSubjDummies   = false;
cfg.P_PERF_MODE      = 'online';     % 'all' / 'online'


% ===== model =====
cfg.MODEL_FAMILY     = 'wyzcond';      % 'cond' / 'coh' / 'blend'/ 'wyz' / 'wyzcoh' / 'wyzcond'
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