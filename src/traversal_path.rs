use std::fmt;
use std::rc::Rc;

use crate::color::Color;
use crate::mv::Move;

// Linked list containing the current path to the root in the minimax tree
// traversal

#[derive(Debug, Clone)]
pub struct TraversalPath {
    list: Rc<TraversalPathElem>,
}

impl TraversalPath {
    pub fn head() -> Self {
        TraversalPath {
            list: Rc::new(TraversalPathElem::Head),
        }
    }

    pub fn append(&self, mv: Move, color: Color) -> Self {
        TraversalPath {
            list: Rc::new(TraversalPathElem::Node(Rc::clone(&self.list), mv, color)),
        }
    }

    pub fn peek(&self) -> Option<(Move, Color)> {
        match &self.list.as_ref() {
            TraversalPathElem::Head => None,
            TraversalPathElem::Node(_, mv, color) => Some((*mv, *color)),
        }
    }

    // More efficient, use this instead of fold_right unless you need items in
    // historical order.
    pub fn fold_left<T>(&self, zero: T, f: fn(accum: T, mv: &Move, color: &Color) -> T) -> T {
        self.list.fold_left(zero, f)
    }

    pub fn fold_right<T>(&self, zero: T, f: fn(accum: T, mv: &Move, color: &Color) -> T) -> T {
        self.list.fold_right(zero, f)
    }

    pub fn len(&self) -> usize {
        self.fold_left(0, |sum, _, _| sum + 1)
    }
}

#[derive(Debug)]
enum TraversalPathElem {
    Head,
    Node(Rc<TraversalPathElem>, Move, Color),
}

impl TraversalPathElem {
    fn fold_right<T>(&self, zero: T, f: fn(accum: T, mv: &Move, color: &Color) -> T) -> T {
        match self {
            TraversalPathElem::Head => zero,
            TraversalPathElem::Node(next, mv, color) => f(next.fold_right(zero, f), mv, color),
        }
    }

    fn fold_left<T>(&self, zero: T, f: fn(accum: T, mv: &Move, color: &Color) -> T) -> T {
        match self {
            TraversalPathElem::Head => zero,
            TraversalPathElem::Node(next, mv, color) => next.fold_left(f(zero, mv, color), f),
        }
    }
}

impl Into<Vec<Move>> for &TraversalPath {
    fn into(self) -> Vec<Move> {
        self.fold_right(Vec::new(), |mut v, mv, _| {
            v.push(*mv);
            v
        })
    }
}

impl fmt::Display for TraversalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.fold_right(String::new(), |mut accum, mv, _| {
            accum.extend(mv.to_string().chars());
            accum.extend(" ".chars());
            accum
        });

        write!(f, "{}", s)
    }
}
