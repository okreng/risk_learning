MAX_STATE = [4,4];
TRIALS = 10000;

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
raw = 0;
% for kk = 1:2
    for ii = 1:MAX_STATE(1)
        for jj = 1:MAX_STATE(2)
            wins = 0;
            start_state = [ii, jj];
            for trial = 1:TRIALS
                state = start_state;
                turn = 1;
                while 1

                    if state(2) < 1
                        wins = wins + turn;
                        break
                    elseif state(1) < 2
                        turn = -turn;
                        if state(2) == MAX_STATE(2)
                            temp = state(1);
                            state(1) = state(2);
                            state(2) = temp;
                        else
                            temp = state(1);
                            state(1) = state(2) + 1;
                            state(2) = temp;
%                         wins = wins - 1;
                        end
                        
                    end
                    if raw
                        state = attack_func_raw(state);
                    else
                        state = attack_func(state);
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
%     if raw
%         disp("Raw state values are");
%         disp(raw_state_values);
%     else
%         disp("Condensed state values are");
%         disp(state_values);
%     end
%     raw = 0;
% end

% disp(state_values-raw_state_values);
% 
 state_values_w_pass = state_values;
 optimal_action = zeros(size(state_values));
 
 
%  update_loop = zeros([1, MAX_STATE(1)*MAX_STATE(2)]);
% for ii = MAX_STATE(1):-1:1
%      while 1 
%          jj = ii - 1; % Assuming square matrix
%          
%          
%          
%          if jj == MAX_STATE(2)
%              break
%          end
%      end
%  end
% 
state_values_w_pass(MAX_STATE(1),MAX_STATE(2)) = max(0, state_values(MAX_STATE(1),MAX_STATE(2)));
for jj = 1:MAX_STATE(2)
    optimal_action(1,jj) = 1;
end
for kk = 1:(MAX_STATE(1)*MAX_STATE(2))
%     for ii = MAX_STATE(1):-1:1
%         for jj = MAX_STATE(2):-1:1
    for ii = 1:MAX_STATE(1)
        for jj = 1:MAX_STATE(2)
            state = [ii,jj];
            if jj < MAX_STATE(2)
                 opponent_arms = jj + 1;
            else
                 opponent_arms = jj;
             end
            opponent_value = state_values_w_pass(opponent_arms,ii);
            player_value = state_values(ii,jj);
            if ~(ii == MAX_STATE(1) && jj == MAX_STATE(2))
                state_values_w_pass(ii,jj) = max(-opponent_value, player_value);
            end
            if (player_value >= (-opponent_value))
                optimal_action(ii,jj) = 1;
            else
                optimal_action(ii,jj) = 0;
            end
        end
    end
end

disp("Attack state values are:");
disp(state_values);
disp("state values are:");
disp(state_values_w_pass);
% disp(state_values_w_pass-state_values);
disp("Optimal actions are:");
disp(optimal_action);