use micrograd::Value;

fn main() {
    let a: Value = 2.0.into();
    let b: Value = (-3.0).into();
    let c: Value = 10.0.into();
    let d = a * b + c;

    dbg!(d);
}
