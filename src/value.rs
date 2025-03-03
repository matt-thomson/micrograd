use std::cell::RefCell;
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Debug)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>,
}

#[derive(Debug)]
struct ValueInner {
    data: f64,
    grad: f64,
    prev: Operation,
}

#[derive(Debug)]
enum Operation {
    Constant,
    Add(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Mul(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Tanh(Rc<RefCell<ValueInner>>),
}

impl From<ValueInner> for Value {
    fn from(inner: ValueInner) -> Self {
        Value {
            inner: Rc::new(RefCell::new(inner)),
        }
    }
}

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            prev: Operation::Constant,
        }
        .into()
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        ValueInner {
            data: self.data() + rhs.data(),
            grad: 0.0,
            prev: Operation::Add(self.inner.clone(), rhs.inner.clone()),
        }
        .into()
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        ValueInner {
            data: self.data() * rhs.data(),
            grad: 0.0,
            prev: Operation::Mul(self.inner.clone(), rhs.inner.clone()),
        }
        .into()
    }
}

impl Value {
    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn tanh(self) -> Value {
        let exp = (self.data() * 2.0).exp();

        ValueInner {
            data: (exp - 1.0) / (exp + 1.0),
            grad: 0.0,
            prev: Operation::Tanh(self.inner.clone()),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_relative_eq;

    use crate::value::Value;

    #[test]
    fn should_calculate_with_values() {
        let a: Value = (2.0).into();
        let b: Value = (-3.0).into();
        let c: Value = (10.0).into();
        let d = a * b + c;
        let f: Value = (-2.0).into();

        let l = d * f;

        assert_eq!(l.data(), -8.0);
    }

    #[test]
    fn should_calculate_tanh() {
        let a: Value = (2.0).into();

        let result = a.tanh();

        assert_float_relative_eq!(result.data(), 0.96402758008);
    }
}
