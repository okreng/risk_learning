MAX_STATE = [6,6];
TRIALS = 100000;

% attack = [1, 0];
% pass = [0, 1];

% for ii = 1:100
%     
%     next_state = attack_func(START_STATE);
% 
%     disp(next_state)
% end

% state = attack_func_raw(MAX_STATE);
raw_state_values = zeros(MAX_STATE);
state_values = zeros(MAX_STATE);
raw = 1;
for kk = 1:2
    for ii = 2:MAX_STATE(1)
        for jj = 1:MAX_STATE(2)
            wins = 0;
            start_state = [ii, jj];
            for trial = 1:TRIALS
                state = start_state;
                while 1
                    if raw
                        state = attack_func_raw(state);
                    else
                        state = attack_func(state);
                    end
                    if state(2) < 1
                        wins = wins + 1;
                        break
                    elseif state(1) < 2
                        break
                    end

                end

            end
            state_value = wins/TRIALS;
            if raw
                raw_state_values(ii,jj) = state_value;
            else
                state_values(ii,jj) = state_value;
            end
        end
    end
    if raw
        disp("Raw state values are");
        disp(raw_state_values);
    else
        disp("Condensed state values are");
        disp(state_values);
    end
    raw = 0;
end

disp(state_values-raw_state_values)
