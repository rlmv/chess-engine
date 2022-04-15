# chess-engine

Work-in-progress UCI compatible chess engine. To play against the engine you will need to use a UCI chess GUI program and point it to the engine binary. I have had good luck with [Cute Chess](https://cutechess.com/), but there are [many other options available.](https://www.chessprogramming.org/UCI)


## Installation

Install the Rust toolchain following instructions [here](https://www.rust-lang.org/tools/install).

This program uses [generators](https://doc.rust-lang.org/beta/unstable-book/language-features/generators.html) which are only available in nightly Rust:

```
rustup toolchain install nightly
rustup default nightly
```

## Release build

Compile a release binary with

```
cargo build --release
```

The binary will be output to `./target/release/chess-engine`.

## Evaluate a position

In addition to the UCI interface, the program provides a command to evaluate a single position from a FEN string:

```
./target/release/chess-engine fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 8
```
The last parameter is the max depth to search.

## Testing

Run tests with:
```
cargo test
```

## Benchmarks

The program contains a small set of corse-grain benchmarks:
```
cargo bench
```
