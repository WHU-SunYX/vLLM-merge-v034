# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share one full-head SSD selection among the workers of a TP group."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)


def _selected_starts(context: dict[str, Any]) -> list[int]:
    """Translate native RPC candidate IDs into logical chunk starts."""
    count = int(context["aissd_candidate_count"][0])
    ids = context["aissd_candidate_chunk_ids"][0, :count].tolist()
    starts = context["aissd_candidate_token_start"][0, :count].tolist()
    lengths = context["aissd_rpc_selected_chunk_lens"]
    selected = context["aissd_rpc_selected_chunk_ids"][0, : int(lengths[0])].tolist()
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError("SSD returned empty or duplicate selected chunk IDs")
    by_id = dict(zip(ids, starts, strict=True))
    return [by_id[chunk_id] for chunk_id in selected]


def _apply_to_local_blocks(context: dict[str, Any], starts: list[int]) -> None:
    """Use this worker's slot mapping to build its sparse attention tables."""
    block_size = int(context["block_size"])
    chunk_size = int(context["chunk_size"])
    slot_map = context["slot_mapping_table"][0]
    offsets = torch.tensor(
        [
            start + offset
            for start in starts
            for offset in range(0, chunk_size, block_size)
        ],
        dtype=torch.long,
        device=slot_map.device,
    )
    slots = slot_map.index_select(0, offsets).tolist()
    if any(slot < 0 or slot % block_size for slot in slots):
        raise RuntimeError("SSD selection maps to an invalid local KV slot")
    remote = [slot // block_size for slot in slots]
    local_count = int(context["host_local_blocks"])
    local_tokens = int(context["host_local_tokens"])
    local = context["local_block_table"][0, :local_count].tolist()
    block_table = remote + local
    selected_table = context["selected_block_table"]
    fa_table = context["fa_block_table"]
    if (
        len(block_table) > fa_table.shape[1]
        or len(remote) > selected_table.shape[1]
        or len(set(block_table)) != len(block_table)
        or any(block < 0 for block in local)
        or (local_tokens + block_size - 1) // block_size != local_count
    ):
        raise RuntimeError("Invalid selected or local block table for SSD Sparse KV TP")

    blocks = torch.tensor(block_table, dtype=fa_table.dtype, device=fa_table.device)
    selected_table[0].fill_(-1)
    selected_table[0, : len(remote)].copy_(blocks[: len(remote)])
    context["selected_block_lens"][0] = len(remote)
    context["selected_ready_flags"][0] = 1
    fa_table[0].zero_()
    fa_table[0, : len(block_table)].copy_(blocks)
    context["fa_block_lens"][0] = len(block_table)
    context["fa_seq_lens"][0] = len(remote) * block_size + local_tokens


def run_tp_ssd_selector(
    tp: "GroupCoordinator",
    query: torch.Tensor,
    context: dict[str, Any],
    select_on_leader: Callable[[torch.Tensor], None],
) -> list[int]:
    """Gather Q, issue the native RPC on TP rank 0 and share logical chunks.

    The existing native op builds rank 0's table. Other ranks compose their
    own tables from the same selected token ranges. A worker error is propagated
    before peers can enter the next model collective.
    """
    # PyTorch process groups can fill collective output buffers from a
    # background thread. Allocate normal tensors for those writes.
    with torch.inference_mode(False):
        full_query = tp.all_gather(query.contiguous(), dim=-2)

    result: tuple[list[int], str | None] | None = None
    if tp.rank_in_group == 0:
        try:
            select_on_leader(full_query)
            result = (_selected_starts(context), None)
        except Exception as exc:
            logger.exception("TP rank 0 failed to select SSD KV chunks")
            result = ([], str(exc))
    with torch.inference_mode(False):
        starts, error = tp.broadcast_object(result)
    if error is not None:
        raise RuntimeError(f"SSD Sparse KV TP selection failed on rank 0: {error}")

    error = None
    chunk_size = int(context["chunk_size"])
    try:
        prefix = int(context["aissd_tp_prefix_tokens"])
        if any(
            start < 0 or start % chunk_size or start + chunk_size > prefix
            for start in starts
        ):
            raise RuntimeError(
                "SSD selection falls outside this worker's cached prefix"
            )
        if tp.rank_in_group != 0:
            _apply_to_local_blocks(context, starts)
    except Exception as exc:
        logger.exception(
            "TP rank %d failed to apply SSD KV selection", tp.rank_in_group
        )
        error = str(exc)
    errors: list[str | None] = [None] * tp.world_size
    with torch.inference_mode(False):
        dist.all_gather_object(errors, error, group=tp.cpu_group)
    for rank, error in enumerate(errors):
        if error is not None:
            raise RuntimeError(
                f"SSD Sparse KV TP block mapping failed on rank {rank}: {error}"
            )
    selected_blocks = len(starts) * (chunk_size // int(context["block_size"]))
    context.update(
        host_selected_blocks=selected_blocks,
        host_selected_blocks_per_req=(selected_blocks,),
        host_selected_chunks=len(starts),
        host_selected_chunks_per_req=(len(starts),),
        host_selected_blocks_source="tp_selected_chunks",
    )
    return starts
