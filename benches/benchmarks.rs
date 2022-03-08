use chess_engine::fen;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn midgame_evaluation(c: &mut Criterion) {
    let board =
        fen::parse("rnbqkbnr/2ppp1pp/1p6/p3PpP1/8/8/PPPP1P1P/RNBQKBNR w KQkq f6 0 5").unwrap();

    c.bench_function("midgame evaluation 4 plies", |b| {
        b.iter(|| board.clone().find_next_move(black_box(4)).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = midgame_evaluation
}
criterion_main!(benches);
