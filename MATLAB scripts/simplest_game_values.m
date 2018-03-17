MAX_STATE = [4,4];
TRIALS = 1000000;

% attack = [1, 0];
% pass = [0, 1];

% for ii = 1:100
%     
%     next_state = attack_func(START_STATE);
% 
%     disp(next_state)
% end


state_values = zeros(4,4);

for ii = 2:4
    for jj = 1:4
        wins = 0;
        start_state = [ii, jj]
        for trial = 1:TRIALS
            state = start_state;
            while 1
                state = attack_func(state);
                if state(2) < 1
                    wins = wins + 1;
                    break
                elseif state(1) < 2
                    break
                end
                
            end
            
        end
        state_value = wins/TRIALS
        state_values(ii,jj) = state_value
    end
end