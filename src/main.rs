use micrograd::Layer;

fn main() {
    let neuron = Layer::new(2, 3);
    let result = neuron.calculate(&[2.0.into(), 3.0.into()]);

    dbg!(result);
}
