use micrograd::{MLP, Value};

fn main() {
    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = [
        [2.0.into(), 3.0.into(), (-1.0).into()],
        [3.0.into(), (-1.0).into(), 0.5.into()],
        [0.5.into(), 1.0.into(), 1.0.into()],
        [1.0.into(), 1.0.into(), 1.0.into()],
    ];

    let ys = [1.0.into(), (-1.0).into(), 1.0.into(), (-1.0).into()];

    let predictions = xs.map(|x| mlp.predict(&x)[0].clone());

    let loss = predictions
        .into_iter()
        .zip(ys)
        .fold(0.0.into(), |acc: Value, (pred, y)| {
            acc + (pred - y).pow(2.0)
        });

    dbg!(loss.value());
}
