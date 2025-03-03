use micrograd::MLP;

fn main() {
    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = [
        [2.0.into(), 3.0.into(), (-1.0).into()],
        [3.0.into(), (-1.0).into(), 0.5.into()],
        [0.5.into(), 1.0.into(), 1.0.into()],
        [1.0.into(), 1.0.into(), 1.0.into()],
    ];

    let ys = [1.0, -1.0, 1.0, -1.0];

    let predictions = xs.map(|x| mlp.calculate(&x)[0].value());
    dbg!(predictions);
}
