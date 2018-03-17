START_STATE = [3,4];
MAX_STATE = [4,4];

attack = [1, 0];
pass = [0, 1];

for ii = 1:100
    
    next_state = attack_func(START_STATE);

    disp(next_state)
end