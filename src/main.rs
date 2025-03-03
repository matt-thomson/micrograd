use micrograd::MLP;

fn main() {
    let mlp = MLP::new(3, &[4, 4, 1]);
    let result = mlp.calculate(&[2.0.into(), 3.0.into(), (-1.0).into()]);

    dbg!(result);
}
