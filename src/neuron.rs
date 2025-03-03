use rand::random;

use crate::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(inputs: usize) -> Self {
        Self {
            weights: (0..inputs).map(|_| random()).collect(),
            bias: random(),
        }
    }
}
