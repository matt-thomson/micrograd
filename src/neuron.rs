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

    pub fn calculate(&self, xs: &[Value]) -> Value {
        self.weights
            .iter()
            .zip(xs)
            .fold(self.bias.clone(), |acc, (w, x)| acc + w.clone() * x.clone())
            .tanh()
    }

    pub fn step(&self, epsilon: f64) {
        self.weights.iter().for_each(|weight| weight.step(epsilon));
        self.bias.step(epsilon);
    }
}
