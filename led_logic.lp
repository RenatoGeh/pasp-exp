0.5?::real(0). 0.5?::real(1). 0.5?::real(2). 0.5?::real(3). 0.5?::real(4). 0.5?::real(5). 0.5?::real(6).

0.5?::observed(0). 0.5?::observed(1). 0.5?::observed(2). 0.5?::observed(3). 
0.5?::observed(4). 0.5?::observed(5). 0.5?::observed(6).

0.5?::observed(0) :- real(0).
0.5?::observed(1) :- real(1).
0.5?::observed(2) :- real(2).
0.5?::observed(3) :- real(3).
0.5?::observed(4) :- real(4).
0.5?::observed(5) :- real(5).
0.5?::observed(6) :- real(6).
0.5?::observed(7) :- real(7).

digit(0) :- real(0), real(1), real(2), not real(3), real(4), real(5), real(6).
digit(1) :- not real(0), not real(1), real(2), not real(3), not real(4), real(5), not real(6).
digit(2) :- real(0), not real(1), real(2), real(3), real(4), not real(5), real(6).
digit(3) :- real(0), not real(1), real(2), real(3), not real(4), real(5), real(6).
digit(4) :- not real(0), real(1), real(2), real(3), real(4), not real(5), not real(6).
digit(5) :- real(0), real(1), not real(2), real(3), not real(4), real(5), real(6).
digit(6) :- real(0), real(1), not real(2), real(3), real(4), real(5), real(6).
digit(7) :- real(0), not real(1), real(2), not real(3), not real(4), real(5), not real(6).
digit(8) :- real(0), real(1), real(2), real(3), real(4), real(5), real(6).
digit(9) :- real(0), real(1), real(2), real(3), not real(4), real(5), real(6).
