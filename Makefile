TIMESTAMP = $(shell date --iso-8601=seconds)
OUTPUT = flamegraph-${TIMESTAMP}.svg
BINARY = ./target/release/chess-engine

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
POSITION_2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0"

release:
	CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release

profile: release
	sudo LOG_LEVEL=INFO LOG_STDOUT=TRUE /home/bo/.cargo/bin/flamegraph --output=${OUTPUT} ${BINARY}  fen ${POSITION_2} 9
	firefox --new-tab file:///${PWD}/${OUTPUT}

run: release
	LOG_LEVEL=INFO LOG_STDOUT=TRUE ${BINARY} fen ${POSITION_2} 10

# Great tutorial: http://sandsoftwaresound.net/perf/perf-tutorial-hot-spots/
perf: release
	sudo LOG_LEVEL=INFO LOG_STDOUT=TRUE perf record -e task-clock,cycles,branch-misses ${BINARY} fen ${POSITION_2} 10
	sudo chown bo:bo perf.data

stat: release
	LOG_LEVEL=INFO LOG_STDOUT=TRUE sudo perf stat -d ${BINARY} fen ${POSITION_2} 9

perf-flamegraph: release
	LOG_LEVEL=INFO LOG_STDOUT=TRUE sudo perf record --call-graph fp -e cpu-clock ${BINARY} fen ${POSITION_2} 6
	sudo chown bo:bo perf.data
	perf script | ../FlameGraph/stackcollapse-perf.pl > out.perf-folded
	../FlameGraph/flamegraph.pl out.perf-folded > perf.svg
	firefox --new-tab file:///${PWD}/perf.svg

dhat: release
	LOG_STDOUT=TRUE	valgrind --tool=dhat ${BINARY} fen ${POSITION_2} 4

logs:
	tail -f -n 100 out.log

cutechess:
	/home/bo/Code/cutechess/projects/gui/cutechess &

perft: release
	${BINARY} perft ${STARTPOS} 6
