% redefine answer by ME
% changed after this script:
% 1. req: left/right correct answer based on the mean motion energy
% 2. Correct: updated newer version as correctness defined

motion_energy = cfg.motion_energy;
req = cfg.req;
Correct = cfg.Correct;
given = cfg.given;
nBins = cfg.nBins;

answer_by_ME = nan(length(motion_energy));
correct_by_ME = nan(length(motion_energy));



nTrials = numel(motion_energy);
if cfg.RTtask
    motion_mat = nan(nTrials, nBins);
    for tr = 1:nTrials
        raw = motion_energy{tr}(:)';
        firstNonZero = find(raw, 1, 'first');
        lastNonZero = find(raw, 1, 'last');  % find last non-zero index
        raw = raw(firstNonZero:lastNonZero);            % trim trailing zeros
        interp_vec = interp1(1:length(raw), raw, linspace(1, length(raw), nBins));
    end 
else
    for tr = 1:nTrials
        motion_mat(tr, :) = motion_energy{tr}(3:38);
    end
end



if cfg.REDEFINE_ANSWER
    % motion energy mean <0 --> answer = 1, motion energy mean > 0 --> answer = 2
    meanme = mean(motion_mat, 2, 'omitnan');
    for i = 1:length(meanme)
        if meanme(i) < 0
            req(i) = 2;
        elseif meanme(i) > 0
            req(i) = 1;
        else
            req(i) = NaN;
        end
    end
    
    for i = 1:length(answer_by_ME)
        if req(i) == given(i)
            Correct(i) = 1;
        else
            Correct(i) = 0;
        end
    end
end 

cfg.motion_energy = motion_energy;
cfg.motion_mat = motion_mat;
cfg.req = req;
cfg.Correct = Correct;