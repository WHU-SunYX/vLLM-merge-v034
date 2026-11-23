// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>

#include "core/registration.h"

// CUDA implementations are registered in the corresponding .cu files.
TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "sparse_ssd_attention(Tensor(a!) out, Tensor query, Tensor kv_cache, "
      "Tensor active_reqs, Tensor req_token_lens, Tensor req_vllm_cached_tokens, "
      "Tensor req_lmcache_cached_tokens, Tensor req_slot_lens, "
      "Tensor slot_mapping_table, Tensor selected_block_table, "
      "Tensor selected_block_lens, Tensor selected_ready_flags, "
      "int block_size, int chunk_size, int top_n_chunks, float scale) -> ()");

  ops.def(
      "sparse_flash_attention(Tensor(a!) out, Tensor query, Tensor kv_cache, "
      "Tensor active_reqs, Tensor req_token_lens, Tensor req_vllm_cached_tokens, "
      "Tensor req_lmcache_cached_tokens, Tensor req_slot_lens, "
      "Tensor slot_mapping_table, Tensor selected_block_table, "
      "Tensor selected_block_lens, Tensor selected_token_lens, "
      "Tensor selected_ready_flags, Tensor(b!) debug_counters, "
      "int block_size, int chunk_size, int top_n_chunks, float scale) -> ()");

  ops.def(
      "aissd_sparse_kv_select("
      "Tensor query, Tensor active_reqs, Tensor req_token_lens, "
      "Tensor req_lmcache_cached_tokens, Tensor aissd_candidate_count, "
      "Tensor aissd_candidate_chunk_ids, Tensor aissd_candidate_block_ids, "
      "Tensor aissd_candidate_block_lens, Tensor aissd_candidate_token_start, "
      "Tensor aissd_candidate_token_end, Tensor aissd_candidate_dtype, "
      "Tensor aissd_candidate_fmt, Tensor aissd_candidate_ndim, "
      "Tensor aissd_candidate_shape, Tensor aissd_candidate_extent_count, "
      "Tensor aissd_candidate_extent_lba, Tensor aissd_candidate_extent_bytes, "
      "Tensor local_block_table, Tensor local_block_lens, "
      "Tensor local_token_lens, "
      "Tensor(a!) selected_block_table, Tensor(b!) selected_block_lens, "
      "Tensor(c!) selected_ready_flags, Tensor(d!) fa_block_table, "
      "Tensor(e!) fa_block_lens, Tensor(f!) fa_seq_lens, "
      "Tensor(g!) aissd_rpc_selected_chunk_ids, "
      "Tensor(h!) aissd_rpc_selected_chunk_lens, "
      "int layer_id, int backend, int num_q_heads, int num_kv_heads, "
      "int head_dim, int chunk_size, int block_size, int top_n_chunks, "
      "int top_m, int score_mode, int manifest_block_size, int timeout_ms) -> ()");
}

REGISTER_EXTENSION(_C_aissd)
