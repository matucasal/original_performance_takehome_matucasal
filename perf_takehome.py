"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD-first kernel:
        - Process VLEN (=8) items at a time with vector ops
        - Use scalar fallback for any remaining tail elements
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        GROUP_VECS = 6  # 6 * VLEN = 48 elements per software-pipelined group

        # Vector scratch registers for grouped processing
        v_idx = [self.alloc_scratch(f"v_idx_{g}", VLEN) for g in range(GROUP_VECS)]
        v_val = [self.alloc_scratch(f"v_val_{g}", VLEN) for g in range(GROUP_VECS)]
        v_node_val = [
            self.alloc_scratch(f"v_node_val_{g}", VLEN) for g in range(GROUP_VECS)
        ]
        v_addr = [self.alloc_scratch(f"v_addr_{g}", VLEN) for g in range(GROUP_VECS)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{g}", VLEN) for g in range(GROUP_VECS)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{g}", VLEN) for g in range(GROUP_VECS)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{g}", VLEN) for g in range(GROUP_VECS)]
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)

        # Scalar temporaries (for base addresses + scalar tail)
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_idx_base = [
            self.alloc_scratch(f"tmp_idx_base_{g}") for g in range(GROUP_VECS)
        ]
        tmp_val_base = [
            self.alloc_scratch(f"tmp_val_base_{g}") for g in range(GROUP_VECS)
        ]

        # Broadcast commonly used scalar constants once.
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        # Broadcast hash constants once.
        hash_vec_consts = {}
        for _, val1, _, _, val3 in HASH_STAGES:
            if val1 not in hash_vec_consts:
                addr = self.alloc_scratch(f"v_const_{val1}", VLEN)
                self.add("valu", ("vbroadcast", addr, self.scratch_const(val1)))
                hash_vec_consts[val1] = addr
            if val3 not in hash_vec_consts:
                addr = self.alloc_scratch(f"v_const_{val3}", VLEN)
                self.add("valu", ("vbroadcast", addr, self.scratch_const(val3)))
                hash_vec_consts[val3] = addr

        def emit_bundle(alu=None, valu=None, load=None, store=None, flow=None):
            instr = {}
            if alu:
                instr["alu"] = alu
            if valu:
                instr["valu"] = valu
            if load:
                instr["load"] = load
            if store:
                instr["store"] = store
            if flow:
                instr["flow"] = flow
            if instr:
                self.instrs.append(instr)

        def emit_valu_chunks(slots, chunk_size=6):
            for j in range(0, len(slots), chunk_size):
                emit_bundle(valu=slots[j : j + chunk_size])

        def emit_group_round_ops(gs):
            # Phase 2: gather address computation for all vectors in this group
            emit_bundle(valu=[("+", v_addr[g], v_forest_base, v_idx[g]) for g in gs])

            # Phase 3: gather loads interleaved by lane pair to keep load slots full
            for lane in range(0, VLEN, 2):
                for g in gs:
                    emit_bundle(
                        load=[
                            ("load_offset", v_node_val[g], v_addr[g], lane),
                            ("load_offset", v_node_val[g], v_addr[g], lane + 1),
                        ]
                    )

            # Phase 4: XOR for all vectors in one cycle
            emit_bundle(valu=[("^", v_val[g], v_val[g], v_node_val[g]) for g in gs])

            # Phase 5: hash stages packed across vectors
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                op13_slots = []
                for g in gs:
                    op13_slots.append((op1, v_tmp1[g], v_val[g], hash_vec_consts[val1]))
                    op13_slots.append((op3, v_tmp2[g], v_val[g], hash_vec_consts[val3]))
                emit_valu_chunks(op13_slots)
                emit_bundle(valu=[(op2, v_val[g], v_tmp1[g], v_tmp2[g]) for g in gs])

            # Phase 6: index update and wrap
            emit_bundle(valu=[("%", v_tmp1[g], v_val[g], v_two) for g in gs])
            emit_bundle(valu=[("==", v_tmp1[g], v_tmp1[g], v_zero) for g in gs])
            emit_bundle(valu=[("-", v_tmp3[g], v_two, v_tmp1[g]) for g in gs])
            emit_bundle(valu=[("*", v_idx[g], v_idx[g], v_two) for g in gs])
            emit_bundle(valu=[("+", v_idx[g], v_idx[g], v_tmp3[g]) for g in gs])
            emit_bundle(valu=[("<", v_tmp1[g], v_idx[g], v_n_nodes) for g in gs])
            emit_bundle(valu=[("*", v_idx[g], v_idx[g], v_tmp1[g]) for g in gs])

        def emit_vec_block_ops(g, write_back=False):
            # Gather node values: v_addr = forest_values_p + v_idx, then lane loads.
            emit_bundle(valu=[("+", v_addr[g], v_forest_base, v_idx[g])])
            emit_bundle(
                load=[
                    ("load_offset", v_node_val[g], v_addr[g], 0),
                    ("load_offset", v_node_val[g], v_addr[g], 1),
                ]
            )
            emit_bundle(
                load=[
                    ("load_offset", v_node_val[g], v_addr[g], 2),
                    ("load_offset", v_node_val[g], v_addr[g], 3),
                ]
            )
            emit_bundle(
                load=[
                    ("load_offset", v_node_val[g], v_addr[g], 4),
                    ("load_offset", v_node_val[g], v_addr[g], 5),
                ]
            )
            emit_bundle(
                load=[
                    ("load_offset", v_node_val[g], v_addr[g], 6),
                    ("load_offset", v_node_val[g], v_addr[g], 7),
                ]
            )

            emit_bundle(valu=[("^", v_val[g], v_val[g], v_node_val[g])])
            for op1, val1, op2, op3, val3 in HASH_STAGES:
                emit_bundle(
                    valu=[
                        (op1, v_tmp1[g], v_val[g], hash_vec_consts[val1]),
                        (op3, v_tmp2[g], v_val[g], hash_vec_consts[val3]),
                    ]
                )
                emit_bundle(valu=[(op2, v_val[g], v_tmp1[g], v_tmp2[g])])

            emit_bundle(valu=[("%", v_tmp1[g], v_val[g], v_two)])
            emit_bundle(valu=[("==", v_tmp1[g], v_tmp1[g], v_zero)])
            emit_bundle(
                valu=[
                    ("-", v_tmp3[g], v_two, v_tmp1[g]),  # branch = 2 - is_even
                    ("*", v_idx[g], v_idx[g], v_two),
                ]
            )
            emit_bundle(valu=[("+", v_idx[g], v_idx[g], v_tmp3[g])])
            emit_bundle(valu=[("<", v_tmp1[g], v_idx[g], v_n_nodes)])
            emit_bundle(
                valu=[("*", v_idx[g], v_idx[g], v_tmp1[g])]  # wrap: idx = idx * is_in_bounds
            )
            if write_back:
                emit_bundle(
                    store=[
                        ("vstore", tmp_idx_base[g], v_idx[g]),
                        ("vstore", tmp_val_base[g], v_val[g]),
                    ]
                )

        vec_batch = (batch_size // VLEN) * VLEN
        group_batch = (vec_batch // (GROUP_VECS * VLEN)) * (GROUP_VECS * VLEN)
        # Grouped processing: load once -> run all rounds -> store once.
        full_gs = list(range(GROUP_VECS))
        for i in range(0, group_batch, GROUP_VECS * VLEN):
            i_consts = [self.scratch_const(i + g * VLEN) for g in range(GROUP_VECS)]

            emit_bundle(
                alu=[
                    slot
                    for g in range(GROUP_VECS)
                    for slot in [
                        ("+", tmp_idx_base[g], self.scratch["inp_indices_p"], i_consts[g]),
                        ("+", tmp_val_base[g], self.scratch["inp_values_p"], i_consts[g]),
                    ]
                ]
            )
            for g in range(GROUP_VECS):
                emit_bundle(
                    load=[
                        ("vload", v_idx[g], tmp_idx_base[g]),
                        ("vload", v_val[g], tmp_val_base[g]),
                    ]
                )

            for _round in range(rounds):
                emit_group_round_ops(full_gs)

            # One writeback per block after all rounds.
            for g in range(GROUP_VECS):
                emit_bundle(
                    store=[
                        ("vstore", tmp_idx_base[g], v_idx[g]),
                        ("vstore", tmp_val_base[g], v_val[g]),
                    ]
                )

        # Leftover full vectors (if batch size not divisible by 48): mini-group them.
        i = group_batch
        while i < vec_batch:
            rem_vecs = (vec_batch - i) // VLEN
            chunk = 2 if rem_vecs >= 2 else 1
            gs = list(range(chunk))
            i_consts = [self.scratch_const(i + g * VLEN) for g in gs]

            emit_bundle(
                alu=[
                    slot
                    for gi, g in enumerate(gs)
                    for slot in [
                        ("+", tmp_idx_base[g], self.scratch["inp_indices_p"], i_consts[gi]),
                        ("+", tmp_val_base[g], self.scratch["inp_values_p"], i_consts[gi]),
                    ]
                ]
            )
            for g in gs:
                emit_bundle(
                    load=[
                        ("vload", v_idx[g], tmp_idx_base[g]),
                        ("vload", v_val[g], tmp_val_base[g]),
                    ]
                )

            for _round in range(rounds):
                emit_group_round_ops(gs)

            for g in gs:
                emit_bundle(
                    store=[
                        ("vstore", tmp_idx_base[g], v_idx[g]),
                        ("vstore", tmp_val_base[g], v_val[g]),
                    ]
                )

            i += chunk * VLEN

        # Scalar tail for non-multiple-of-VLEN batch sizes: load once, run rounds, store once.
        for i in range(vec_batch, batch_size):
            i_const = self.scratch_const(i)
            emit_bundle(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], i_const)])
            emit_bundle(load=[("load", tmp_idx, tmp_addr)])
            emit_bundle(alu=[("+", tmp_addr, self.scratch["inp_values_p"], i_const)])
            emit_bundle(load=[("load", tmp_val, tmp_addr)])
            for _round in range(rounds):
                emit_bundle(alu=[("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)])
                emit_bundle(load=[("load", tmp_node_val, tmp_addr)])
                emit_bundle(alu=[("^", tmp_val, tmp_val, tmp_node_val)])
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    emit_bundle(
                        alu=[
                            (op1, tmp1, tmp_val, self.scratch_const(val1)),
                            (op3, tmp2, tmp_val, self.scratch_const(val3)),
                        ]
                    )
                    emit_bundle(alu=[(op2, tmp_val, tmp1, tmp2)])
                emit_bundle(alu=[("%", tmp1, tmp_val, two_const)])
                emit_bundle(alu=[("==", tmp1, tmp1, zero_const)])
                emit_bundle(alu=[("-", tmp3, two_const, tmp1)])
                emit_bundle(alu=[("*", tmp_idx, tmp_idx, two_const)])
                emit_bundle(alu=[("+", tmp_idx, tmp_idx, tmp3)])
                emit_bundle(alu=[("<", tmp1, tmp_idx, self.scratch["n_nodes"])])
                emit_bundle(alu=[("*", tmp_idx, tmp_idx, tmp1)])
            emit_bundle(alu=[("+", tmp_addr, self.scratch["inp_indices_p"], i_const)])
            emit_bundle(store=[("store", tmp_addr, tmp_idx)])
            emit_bundle(alu=[("+", tmp_addr, self.scratch["inp_values_p"], i_const)])
            emit_bundle(store=[("store", tmp_addr, tmp_val)])

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
