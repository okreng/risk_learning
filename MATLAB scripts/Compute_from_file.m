MAX_STATE = [4,4];

pWin = zeros(MAX_STATE) -1;
sArmy = zeros(MAX_STATE) -1;
sDef = zeros(MAX_STATE) -1;
A = zeros(MAX_STATE);
B = zeros(MAX_STATE);

for a = 2:MAX_STATE(1)
    disp(a);
    A(a,b) = a;
    for b = 1:MAX_STATE(2)-1
        [pWin, sArmy, sDef] = probWin(a, b, pWin, sArmy, sDef);
        B(a,b) = b
    end
end

state_values = pWin;
disp(state_values);
disp(sArmy);
disp(sDef);
disp(A);
disp(B);