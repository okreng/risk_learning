function [next_state] = attack_func(state)
%UNTITLED2 Attack move in Risk
%   Detailed explanation goes here
% 
% player = state[1];
% enemy = state[2];

next_state = state;

choice = rand();

if (state(1) == 4)
    if (state(2) == 1)
        arms = 1;
        if choice < (855/1296)
            result = 1;
        else
            result = -1;
        end
    else
        arms = 2;
        if choice < (2890/7776)
            result = 1;
        elseif choice < (5501/7776)
            result = 0;
        else
            result = -1;
        end
    end

elseif state(1) == 3
    if (state(2) == 1)
        arms = 1;
        if choice < (125/216)
            result = 1;
        else
            result = -1;
        end
    else
        arms = 2;
        if choice < (295/1296)
            result = 1;
        elseif choice < (715/1295)
            result = 0;
        else
            result = -1;
        end
    end

elseif (state(1)) == 2
    arms = 1;
    if state(2) == 1
        if choice < (15/36)
            result = 1;
        else
            result = -1;
        end
    else
        if choice < (55/216)
            result = 1;
        else
            result = -1;
        end
    end
    
else 
    disp("Can't attack with 1 army");
end

if arms == 2
    if result == 1
        next_state(2) = state(2) - 2;
    elseif result == 0
        next_state(1) = state(1) - 1;
        next_state(2) = state(2) - 1;
    else
        next_state(1) = state(1) - 2;
    end
else
    if result == 1
        next_state(2) = state(2) - 1;
    else
        next_state(1) = state(1) - 1;
    end
end
  

return

end

