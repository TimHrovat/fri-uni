type state = Q of int | Final
type direction = Left | Right | Stay
type cell = Blank | Zero | One
type tape = { left : cell list; right : cell list }
type program = ((state * cell) * (state * cell * direction)) list
let copy : program =
[
((Q 0, Blank), (Final, Blank, Stay));
((Q 0, One), (Q 2, Blank, Right));
((Q 2, Blank), (Q 3, Blank, Right));
((Q 2, One), (Q 2, One, Right));
((Q 3, Blank), (Q 4, One, Left));
((Q 3, One), (Q 3, One, Right));
((Q 4, Blank), (Q 5, Blank, Left));
((Q 4, One), (Q 4, One, Left));
((Q 5, Blank), (Q 0, One, Right));
((Q 5, One), (Q 5, One, Left));
]
let parity : program =
[
((Q 0, Zero), (Q 0, Zero, Right));
((Q 0, One), (Q 1, One, Right));
((Q 0, Blank), (Final, Zero, Stay));
((Q 1, Zero), (Q 1, Zero, Right));
((Q 1, One), (Q 0, One, Right));
((Q 1, Blank), (Final, One, Stay));
]
let plus1 : program =
[
((Q 0, One), (Q 0, One, Right));
((Q 0, Blank), (Final, One, Stay));
]
let action direction tape : tape =
let move_left = function
| { left = []; right } -> { left = []; right = Blank :: right }
| { left = h :: t; right } -> { left = t; right = h :: right }
in
let move_right = function
| { left; right = [] } -> { left = Blank :: left; right = [] }
| { left; right = h :: t } -> { left = h :: left; right = t }
in
match direction with
| Stay -> tape
| Left -> move_left tape
| Right -> move_right tape

let read (t : tape) : cell =
  match t.right with
  | [] -> Blank
  | h :: _ -> h

let write (c : cell) (t : tape) : tape =
  match t.right with
  | [] -> { t with right = [c] }
  | _ :: rest -> { t with right = c :: rest }

let step (prog: program) (st: state) (t: tape) : state * tape =
    let current = read t in
    let (next_state, write_cell, move_dir) = List.assoc (st, current) prog in
    let updated_tape = write write _cell t in
    let moved_tape = action move_dir updated_tape in
    (next_state, moved_tape)
