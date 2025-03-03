use micrograd::Value;

fn main() {
    let x1: Value = (2.0).into();
    let x2: Value = (0.0).into();
    let w1: Value = (-3.0).into();
    let w2: Value = (1.0).into();
    let b: Value = (6.88137358701957432).into();

    let n = (x1 * w1) + (x2 * w2) + b;
    let o = n.tanh();

    o.backward();

    dbg!(o);
}
