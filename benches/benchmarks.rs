use chess_engine::fen;

use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};
use std::time::Duration;

fn midgame_evaluation(c: &mut Criterion) {
    let board =
        fen::parse("rnbqkbnr/2ppp1pp/1p6/p3PpP1/8/8/PPPP1P1P/RNBQKBNR w KQkq f6 0 5").unwrap();

    c.bench_function("midgame evaluation 4 plies", |b| {
        b.iter(|| board.clone().find_next_move(black_box(4)).unwrap())
    });
}

fn midgame_evaluation_6_plies(c: &mut Criterion) {
    let board =
        fen::parse("rnbqkbnr/2ppp1pp/1p6/p3PpP1/8/8/PPPP1P1P/RNBQKBNR w KQkq f6 0 5").unwrap();

    c.bench_function("midgame evaluation 6 plies", |b| {
        b.iter(|| board.clone().find_next_move(black_box(6)).unwrap())
    });
}

fn midgame_evaluation_8_plies(c: &mut Criterion) {
    let mut group = c.benchmark_group("slow");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let board =
        fen::parse("rnbqkbnr/2ppp1pp/1p6/p3PpP1/8/8/PPPP1P1P/RNBQKBNR w KQkq f6 0 5").unwrap();

    group.bench_function("midgame evaluation 8 plies", |b| {
        b.iter(|| board.clone().find_next_move(black_box(8)).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = midgame_evaluation,midgame_evaluation_6_plies,midgame_evaluation_8_plies
}
criterion_main!(benches);
