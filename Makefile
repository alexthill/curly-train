NAME=curly-train
BIN_NAME=scop
CARGO=/root/.cargo/bin/cargo
CONTAINER=rust-rust-1
TARGET=$(HOME)/goinfre/rust_root/target/debug
TARGET_RELEASE=$(HOME)/goinfre/rust_root/target/release

c: check
check:
	@docker exec --tty --workdir /src/$(NAME) $(CONTAINER) $(CARGO) check

b: build
build:
	@docker exec --tty --workdir /src/$(NAME) $(CONTAINER) $(CARGO) build

br: build_release
build_release:
	@docker exec --tty --workdir /src/$(NAME) $(CONTAINER) $(CARGO) build --release

r: run
run: build
	RUST_LOG=debug $(TARGET)/$(BIN_NAME)

rr: run_release
run_release: build_release
	@RUST_LOG=debug $(TARGET_RELEASE)/$(BIN_NAME)

t: test
test:
	@docker exec --tty --workdir /src/$(NAME) $(CONTAINER) $(CARGO) test

clear:
	@docker exec --tty --workdir /src/$(NAME) $(CONTAINER) $(CARGO) clean
