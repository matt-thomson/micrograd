use crate::{Neuron, Value};

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            neurons: (0..outputs).map(|_| Neuron::new(inputs)).collect(),
        }
    }

    pub fn calculate(&self, xs: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.calculate(xs))
            .collect()
    }

    pub fn step(&self, epsilon: f64) {
        self.neurons.iter().for_each(|neuron| neuron.step(epsilon));
    }
}
