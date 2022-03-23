TIMESTAMP = $(shell date --iso-8601=seconds)
OUTPUT = flamegraph-${TIMESTAMP}.svg


release:
	CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release

profile: release
	sudo LOG_LEVEL=INFO LOG_STDOUT=TRUE /home/bo/.cargo/bin/flamegraph --output=${OUTPUT} ./target/release/chess-engine  fen "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3" 8
	firefox --new-tab file:///${PWD}/${OUTPUT}

# Great tutorial: http://sandsoftwaresound.net/perf/perf-tutorial-hot-spots/
perf: release
	LOG_LEVEL=INFO LOG_STDOUT=TRUE sudo perf record -e task-clock,cpu-clock,faults,cache-misses ./target/release/chess-engine  fen "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3" 8
	sudo chown bo:bo perf.data

perf-flamegraph: release
	LOG_LEVEL=INFO LOG_STDOUT=TRUE sudo perf record --call-graph fp -e cpu-clock ./target/release/chess-engine fen "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3" 6
	sudo chown bo:bo perf.data
	perf script | ../FlameGraph/stackcollapse-perf.pl > out.perf-folded
	../FlameGraph/flamegraph.pl out.perf-folded > perf.svg
	firefox --new-tab file:///${PWD}/perf.svg

dhat: release
	LOG_STDOUT=TRUE	valgrind --tool=dhat ./target/release/chess-engine fen "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3" 4

logs:
	tail -f -n 100 out.log

cutechess:
	/home/bo/Code/cutechess/projects/gui/cutechess &
