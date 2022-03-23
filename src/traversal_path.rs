use std::fmt;
use std::rc::Rc;

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

    pub fn append(&self, mv: Move) -> Self {
        TraversalPath {
            list: Rc::new(TraversalPathElem::Node(Rc::clone(&self.list), mv)),
        }
    }

    pub fn fold_left<T>(&self, zero: T, f: fn(accum: T, mv: &Move) -> T) -> T {
        self.list.fold_left(zero, f)
    }

    pub fn len(&self) -> usize {
        self.fold_left(0, |sum, _| sum + 1)
    }
}

#[derive(Debug)]
enum TraversalPathElem {
    Head,
    Node(Rc<TraversalPathElem>, Move),
}

impl TraversalPathElem {
    fn fold_left<T>(&self, zero: T, f: fn(accum: T, mv: &Move) -> T) -> T {
        match self {
            TraversalPathElem::Head => zero,
            TraversalPathElem::Node(next, mv) => f(next.fold_left(zero, f), mv),
        }
    }
}

impl Into<Vec<Move>> for &TraversalPath {
    fn into(self) -> Vec<Move> {
        self.fold_left(Vec::new(), |mut v, mv| {
            v.push(*mv);
            v
        })
    }
}

impl fmt::Display for TraversalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.fold_left(String::new(), |mut accum, mv| {
            accum.extend(mv.to_string().chars());
            accum.extend(" ".chars());
            accum
        });

        write!(f, "{}", s)
    }
}
