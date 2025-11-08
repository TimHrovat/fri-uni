dedup([], []).
dedup([X|Xs], D) :-
    count_same(X, Xs, N, Rest),
    (   N mod 2 =:= 1
    ->  D = [X|D1]
    ;   D = D1
    ),
    dedup(Rest, D1).

% count_same(X, List, Count, Rest):
% prešteje koliko zaporednih X je v List + X (torej skupaj), poda Count in preostanek Rest
count_same(X, [], 1, []).
count_same(X, [X|Xs], N, Rest) :-
    count_same(X, Xs, N1, Rest),
    N is N1 + 1.
count_same(X, [Y|Ys], 1, [Y|Ys]) :-
    X \= Y.




