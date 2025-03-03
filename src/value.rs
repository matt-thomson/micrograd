use std::ops::{Add, Mul};

#[derive(Debug)]
pub struct Value {
    data: f32,
    prev: Box<Children>,
}

#[derive(Debug)]
enum Children {
    None,
    Add(Value, Value),
    Mul(Value, Value),
}

impl From<f32> for Value {
    fn from(data: f32) -> Self {
        Value {
            data,
            prev: Box::new(Children::None),
        }
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            prev: Box::new(Children::Add(self, rhs)),
        }
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            prev: Box::new(Children::Mul(self, rhs)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Value;

    #[test]
    fn should_calculate_with_values() {
        let a: Value = (2.0).into();
        let b: Value = (-3.0).into();
        let c: Value = (10.0).into();
        let d = a * b + c;
        let f: Value = (-2.0).into();

        let l = d * f;

        assert_eq!(l.data, -8.0);
    }
}
