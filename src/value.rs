use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

use rand::distr::{Distribution, StandardUniform};

#[derive(Debug, Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>,
}

#[derive(Debug)]
struct ValueInner {
    value: f64,
    gradient: f64,
    operation: Operation,
}

#[derive(Debug)]
enum Operation {
    Constant,
    Add(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Sub(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Mul(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Pow(Rc<RefCell<ValueInner>>, f64),
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
            value: data,
            gradient: 0.0,
            operation: Operation::Constant,
        }
        .into()
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        ValueInner {
            value: self.value() + rhs.value(),
            gradient: 0.0,
            operation: Operation::Add(self.inner.clone(), rhs.inner.clone()),
        }
        .into()
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Self::Output {
        ValueInner {
            value: self.value() - rhs.value(),
            gradient: 0.0,
            operation: Operation::Sub(self.inner.clone(), rhs.inner.clone()),
        }
        .into()
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        ValueInner {
            value: self.value() * rhs.value(),
            gradient: 0.0,
            operation: Operation::Mul(self.inner.clone(), rhs.inner.clone()),
        }
        .into()
    }
}

impl Distribution<Value> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Value {
        let value: f64 = rng.random();
        (value * 2.0 - 1.0).into()
    }
}

impl Value {
    pub fn value(&self) -> f64 {
        self.inner.borrow().value
    }

    pub fn pow(self, exponent: f64) -> Value {
        ValueInner {
            value: self.value().powf(exponent),
            gradient: 0.0,
            operation: Operation::Pow(self.inner.clone(), exponent),
        }
        .into()
    }

    pub fn tanh(self) -> Value {
        let exp = (self.value() * 2.0).exp();

        ValueInner {
            value: (exp - 1.0) / (exp + 1.0),
            gradient: 0.0,
            operation: Operation::Tanh(self.inner.clone()),
        }
        .into()
    }

    pub fn backward(&self) {
        self.inner.borrow_mut().gradient = 1.0;
        self.inner.borrow().backward();
    }
}

impl ValueInner {
    fn backward(&self) {
        match &self.operation {
            Operation::Constant => {}
            Operation::Add(left, right) => {
                left.borrow_mut().gradient += self.gradient;
                right.borrow_mut().gradient += self.gradient;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Sub(left, right) => {
                left.borrow_mut().gradient += self.gradient;
                right.borrow_mut().gradient -= self.gradient;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Mul(left, right) => {
                left.borrow_mut().gradient += self.gradient * right.borrow().value;
                right.borrow_mut().gradient += self.gradient * left.borrow().value;

                left.borrow().backward();
                right.borrow().backward();
            }
            Operation::Pow(base, exponent) => {
                let base_value = base.borrow().value;

                base.borrow_mut().gradient +=
                    self.gradient * exponent * base_value.powf(exponent - 1.0);

                base.borrow().backward();
            }
            Operation::Tanh(value) => {
                value.borrow_mut().gradient += self.gradient * (1.0 - self.value.powi(2));

                value.borrow().backward();
            }
        }
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

        assert_eq!(l.value(), -8.0);
    }

    #[test]
    fn should_calculate_tanh() {
        let a: Value = (2.0).into();

        let result = a.tanh();

        assert_float_relative_eq!(result.value(), 0.96402758008);
    }

    #[test]
    fn should_backpropagate() {
        let x1: Value = (2.0).into();
        let x2: Value = (0.0).into();
        let w1: Value = (-3.0).into();
        let w2: Value = (1.0).into();
        let b: Value = (6.88137358701957432).into();

        let n = (x1.clone() * w1.clone()) + (x2.clone() * w2.clone()) + b.clone();
        let o = n.clone().tanh();

        o.backward();

        assert_float_relative_eq!(n.inner.borrow().gradient, 0.5);
        assert_float_relative_eq!(w1.inner.borrow().gradient, 1.0);
        assert_float_relative_eq!(w2.inner.borrow().gradient, 0.0);
    }

    #[test]
    fn should_handle_same_node_twice() {
        let a: Value = 3.0.into();
        let b = a.clone() + a.clone();

        b.backward();

        assert_float_relative_eq!(a.inner.borrow().gradient, 2.0);
    }

    #[test]
    fn should_differentiate_sub() {
        let a: Value = 2.0.into();
        let b: Value = 3.0.into();

        let result = a.clone() - b.clone();
        result.backward();

        assert_float_relative_eq!(a.inner.borrow().gradient, 1.0);
        assert_float_relative_eq!(b.inner.borrow().gradient, -1.0);
    }

    #[test]
    fn should_differentiate_pow() {
        let a: Value = 2.0.into();

        let result = a.clone().pow(3.0);
        result.backward();

        assert_float_relative_eq!(a.inner.borrow().gradient, 12.0);
    }
}
