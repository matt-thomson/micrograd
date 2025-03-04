use crate::{Layer, Value};

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(inputs: usize, sizes: &[usize]) -> Self {
        Self {
            layers: [&[inputs], sizes]
                .concat()
                .windows(2)
                .map(|w| Layer::new(w[0], w[1]))
                .collect(),
        }
    }

    pub fn predict(&self, xs: &[Value]) -> Vec<Value> {
        let first = self.layers[0].calculate(xs);

        self.layers[1..]
            .iter()
            .fold(first, |acc, layer| layer.calculate(&acc))
    }

    pub fn step(&self, epsilon: f64) {
        self.layers.iter().for_each(|layer| layer.step(epsilon));
    }
}
