% redefine answer by ME
% changed after this script:
% 1. req: left/right correct answer based on the mean motion energy
% 2. Correct: updated req == given

motion_energy = cfg.motion_energy;
req = cfg.req;
Correct = cfg.Correct;
given = cfg.given;

answer_by_ME = nan(length(motion_energy));
correct_by_ME = nan(length(motion_energy));

nTrials = numel(motion_energy);
motion_mat = nan(nTrials, 36);

for tr = 1:nTrials
    motion_mat(tr, :) = motion_energy{tr}(3:38);
end

% motion energy mean <0 --> answer = 1, motion energy mean > 0 --> answer = 2
meanme = mean(motion_mat, 2, 'omitnan');
for i = 1:length(meanme)
    if meanme(i) < 0
        req(i) = 2;
    elseif meanme(i) ==0
        fprint('a trial ==0');
    else
        req(i) = 1;
    end
end

for i = 1:length(answer_by_ME)
    if req(i) == given(i)
        Correct(i) = 1;
    else
        Correct(i) = 0;
    end
end

cfg.motion_energy = motion_energy;
cfg.motion_mat = motion_mat;
cfg.req = req;
cfg.Correct = Correct;