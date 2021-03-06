function [next_state] = attack_func_raw(state)
%UNTITLED2 Attack move in Risk
%   Detailed explanation goes here
% 
% player = state[1];
% enemy = state[2];

next_state = state;

player_dice = min(3,state(1)-1);
enemy_dice = min(2, state(2));

arms = min(enemy_dice, player_dice);

player_rolls = randi([1,6],[1,player_dice]);
enemy_rolls = randi([1,6],[1,enemy_dice]);

player_rolls = sort(player_rolls, 'descend');
enemy_rolls = sort(enemy_rolls, 'descend');

engagement = zeros(2, arms);
engagement(1,:) = player_rolls(1:arms);
engagement(2,:) = enemy_rolls(1:arms);

if arms == 1
    result = ((engagement(1,1) > engagement(2,1)));
else
    match1 = (engagement(1,1) > engagement(2,1));
    match2 = (engagement(1,2) > engagement(2,2));
    result = (match1 + match2 - 1);
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

