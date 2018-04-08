% function probWin
%
% Yi Chai
% March 27, 2008
%
% for CME305 project 2008
%
% this function recursively computes the entries in the pWin and sArmy
% matrices until the element in the last row and last column is evaluated
%
% this is used as a subroutined called by masterWin function in evaluating
% the battle between an attacking territory and a defending territory in
% the board game Risk
%
% Inputs:
% a the number of troops in the attacking territory
% b the number of troops in the defending territory
% pWin a matrix where the i,j element denote the probability of
% successful attack given i attackers and j defenders
% sArmy a matrix where the i,j element denote the expected number of
% surviving troops after successful attack given i attackers and j
% defenders
% sDef a matrix where the i,j element denote the expected number of
% surviving defender troops after the attack failed, given i
% attackers and j defenders
%
% Outputs:
% pWin updated probability matrix
% sArmy updated expected number of survival troops matrix
% sDef updated expected number of survival defender troops matrix
%
% Assumptions of rules:
% max allowed number of dices are always rolled when possible for each side
% attacker receives up to 3 dices, defender receives up to 2 dices
% the attacking territory fails if only 1 troop remains
% the defending territory fails if zero troop remains
function [pWin, sArmy, sDef] = probWin(a, d, pWin, sArmy, sDef)
% if the particular scenario (a, d) is already computed previously
% do not compute again
if (pWin(a, d + 1) >= 0)
 return;
end
% initializes the probability matrices for different dies
% probability table obtained from:
% http://en.wikipedia.org/wiki/Risk_game
A = zeros(3, 2);
B = zeros(3, 2);
C = zeros(3, 2);
%Attacker Wins
A(1,1)=15/36;
A(1,2)=55/216;
A(2,1)=125/216;
A(2,2)=295/1296;
A(3,1)=855/1296;
A(3,2)=2890/7776;
%Defender Wins
B(1,1)=21/36;
B(1,2)=161/216;
B(2,1)=91/216;
B(2,2)=581/1296;
B(3,1)=441/1296;
B(3,2)=2275/7776;
%Both Win One
C(1,1)=0;
C(1,2)=0;
C(2,1)=0;
C(2,2)=420/1296;
C(3,1)=0;
C(3,2)=2611/7776;
% compute how many dices attacker is rolling
if (a > 3)
 aDice = 3;
elseif (a == 3)
 aDice = 2;
elseif (a == 2)
 aDice = 1;
else
 % in this case, attacker loses with only 1 troop remaining
 pWin(a, d + 1) = 0;
 sArmy(a, d + 1) = 1;
 sDef(a, d + 1) = d;
 return;
end
% compute how many dices defender is rolling
if (d > 1)
 dDice = 2;
elseif (d == 1)
 dDice = 1;
else
 % in this case, defender loses with zero troop remaining
 pWin(a, d + 1) = 1;
 sArmy(a, d + 1) = a;
 sDef(a, d + 1) = d;
 return;
end
% compute 3 sub-components of the probability tree, corresponding to the 3
% cases where attcker wins dice round, defender wins dice round, or each
% side loses one army
% update the appropriate elements in the matrices recursively
[pWin, sArmy, sDef] = probWin(a, d - dDice, pWin, sArmy, sDef);
[pWin, sArmy, sDef] = probWin(max(1, a - dDice), d, pWin, sArmy, sDef);
[pWin, sArmy, sDef] = probWin(a - 1, d - 1, pWin, sArmy, sDef);
% update the matrix entries of this particular instance as a weighted
% average of the 3 sub-components, with the attacker winning probability of
% each scenario as the weights
pWin(a, d + 1) = A(aDice, dDice)*pWin(a, d - dDice + 1)...
 + B(aDice, dDice)*pWin(max(1, a - dDice), d + 1)...
 + C(aDice, dDice)*pWin(a - 1, d);
sArmy(a, d + 1) = A(aDice, dDice)*sArmy(a, d - dDice + 1)...
 + B(aDice, dDice)*sArmy(max(1, a - dDice), d + 1)...
 + C(aDice, dDice)*sArmy(a - 1, d);
sDef(a, d + 1) = A(aDice, dDice)*sDef(a, d - dDice + 1)...
 + B(aDice, dDice)*sDef(max(1, a - dDice), d + 1)...
 + C(aDice, dDice)*sDef(a - 1, d);