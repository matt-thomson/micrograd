use micrograd::Value;

fn main() {
    let h = 0.001;

    let l1 = calculate_l(0.0);
    let l2 = calculate_l(h);

    let grad = (l2 - l1) / h;
    dbg!(grad);
}

fn calculate_l(h: f32) -> f32 {
    let a: Value = (2.0 + h).into();
    let b: Value = (-3.0).into();
    let c: Value = 10.0.into();
    let d = a * b + c;
    let f: Value = (-2.0).into();

    let l = d * f;

    l.data
}
