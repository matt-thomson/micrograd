use micrograd::Neuron;

fn main() {
    let neuron = Neuron::new(2);
    let result = neuron.calculate(vec![2.0.into(), 3.0.into()]);

    dbg!(result);
}
