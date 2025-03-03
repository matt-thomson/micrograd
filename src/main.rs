use micrograd::Value;

fn main() {
    let h = 0.00001;

    let n1 = calculate_neuron(0.0);
    let n2 = calculate_neuron(h);

    let grad = (n2 - n1) / h;
    dbg!(grad);
}

fn calculate_neuron(h: f64) -> f64 {
    let x1: Value = (2.0).into();
    let x2: Value = (0.0).into();
    let w1: Value = (-3.0 + h).into();
    let w2: Value = (1.0).into();
    let b: Value = (6.7).into();

    let n = (x1 * w1) + (x2 * w2) + b;
    let o = n.tanh();

    o.data
}
