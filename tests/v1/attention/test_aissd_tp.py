# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.attention.backends.aissd_tp import run_tp_ssd_selector

pytestmark = pytest.mark.cpu_test


def _run_workers(rank: int, tp_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        world_size=tp_size,
        rank=rank,
        timeout=timedelta(seconds=30),
    )
    tp = GroupCoordinator(
        group_ranks=[list(range(tp_size))],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_device_communicator=True,
        use_message_queue_broadcaster=True,
        group_name="aissd_tp_test",
    )
    try:
        _check_worker_selection(tp, rank, tp_size)
    finally:
        tp.destroy()
        dist.destroy_process_group()


@torch.inference_mode()
def _check_worker_selection(tp: GroupCoordinator, rank: int, tp_size: int) -> None:
    block_offset = rank * 32
    prefix_blocks = torch.tensor([[11, 3, 17, 2, 10, 4]]) + block_offset
    slot_map = (prefix_blocks.unsqueeze(-1) * 2 + torch.arange(2)).reshape(1, -1)
    tail = torch.tensor([[21, 22]], dtype=torch.int32) + block_offset
    fa_blocks = torch.tensor([[10, 4, 11, 3, 21, 22]]) + block_offset
    context = {
        "block_size": 2,
        "chunk_size": 4,
        "aissd_tp_prefix_tokens": 12,
        "slot_mapping_table": slot_map,
        "local_block_table": tail,
        "host_local_blocks": 2,
        "host_local_tokens": 3,
        "selected_block_table": torch.full((1, 6), -1, dtype=torch.int32),
        "selected_block_lens": torch.zeros(1, dtype=torch.int32),
        "selected_ready_flags": torch.zeros(1, dtype=torch.int32),
        "fa_block_table": torch.zeros((1, 6), dtype=torch.int32),
        "fa_block_lens": torch.zeros(1, dtype=torch.int32),
        "fa_seq_lens": torch.zeros(1, dtype=torch.int32),
    }
    calls = 0

    def fake_ssd_rpc(full_query: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        assert rank == 0
        assert torch.equal(
            full_query,
            torch.arange(tp_size * 4, dtype=torch.bfloat16).reshape(1, -1, 2),
        )
        context["aissd_candidate_count"] = torch.tensor([3], dtype=torch.int32)
        context["aissd_candidate_chunk_ids"] = torch.tensor([[7, 42, 8]])
        context["aissd_candidate_token_start"] = torch.tensor([[0, 4, 8]])
        context["aissd_rpc_selected_chunk_ids"] = torch.tensor([[8, 7]])
        context["aissd_rpc_selected_chunk_lens"] = torch.tensor([2])
        context["selected_block_table"][0, :4] = fa_blocks[0, :4]
        context["selected_block_lens"].fill_(4)
        context["selected_ready_flags"].fill_(1)
        context["fa_block_table"].copy_(fa_blocks)
        context["fa_block_lens"].fill_(6)
        context["fa_seq_lens"].fill_(11)

    query = torch.arange(rank * 4, rank * 4 + 4, dtype=torch.bfloat16).reshape(1, 2, 2)
    assert run_tp_ssd_selector(tp, query, context, fake_ssd_rpc) == [8, 0]
    assert calls == (1 if rank == 0 else 0)
    assert torch.equal(context["fa_block_table"], fa_blocks)
    assert torch.equal(context["selected_block_table"][0, :4], fa_blocks[0, :4])
    assert context["selected_ready_flags"].item() == 1
    assert context["selected_block_lens"].item() == 4
    assert context["fa_block_lens"].item() == 6
    assert context["fa_seq_lens"].item() == 11

    def failed_ssd_rpc(_: torch.Tensor) -> None:
        raise OSError("simulated SSD timeout")

    with pytest.raises(RuntimeError, match="rank 0: simulated SSD timeout"):
        run_tp_ssd_selector(tp, query, context, failed_ssd_rpc)

    if rank == 1:
        slot_map[0, 8] = -1
    with pytest.raises(RuntimeError, match="block mapping failed on rank 1"):
        run_tp_ssd_selector(tp, query, context, fake_ssd_rpc)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_ssd_sparse_kv_tp(tp_size: int, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    mp.spawn(
        _run_workers,
        args=(tp_size, f"file://{tmp_path / 'tp_rendezvous'}"),
        nprocs=tp_size,
    )
