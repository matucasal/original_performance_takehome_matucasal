# Submission Summary

## What I changed (only the experiments that helped)

### Step 1) Move from scalar to SIMD (8 lanes)
- I changed the hot loop from processing 1 item at a time to 8 items at a time (`VLEN=8`).
- I used `vload`, `valu`, and `vstore` for the main path.
- I kept a scalar tail for leftovers.

Result:
- Around `147734` cycles -> around `15437` cycles.

### Step 2) Group vectors and run work in phases
- Instead of handling one vector block at a time, I grouped several vectors together.
- I emitted instructions in phases: address, gather loads, xor, hash ops, index update, store.
- This made the engines busier and reduced bubbles.

Result:
- Around `15437` -> `8014` -> `7182` -> `5654` (with better grouping).

### Step 3) Replace flow-style decisions with arithmetic
- I removed decision patterns that put pressure on the flow path.
- I used arithmetic equivalents, for example:
  - `branch = 2 - is_even`
  - wrapping index with multiply-by-mask.

Result:
- Around `8014` -> `7182`.

### Step 4) Load once / store once across all rounds
- For each vector group, I load `idx` and `val` once.
- I run all rounds while values stay in vector scratch.
- I store once at the end.

Result:
- Around `7182` -> `6042`.

### Step 5) Increase group width and chunk VALU safely
- I increased `GROUP_VECS` and packed more hash work per group.
- I chunked VALU emissions to respect slot limits.
- Best stable group size became `GROUP_VECS=6`.

Result:
- `5654` -> `4997`.

### Step 6) Improve leftover vector handling
- For remaining full vectors, I stopped falling back to single-vector processing.
- I process leftovers in mini-groups (best/tied with `chunk=2`).

Result:
- `4997` -> `4692`.

## Final good configuration
- `GROUP_VECS = 6`
- Grouped phased processing
- Arithmetic branch/wrap updates
- Load-once/store-once across rounds
- Mini-group leftover handling

Best measured result in this optimization path:
- `4692` cycles (from scalar baseline `147734`).
