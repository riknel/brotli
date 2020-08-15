/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

#include "./state.h"

#include <stdlib.h>  /* free, malloc */

#include <brotli/types.h>
#include "./huffman.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

void InitBlockSplitStored(BrotliDecoderState* s,
                          BlockSplitFromDecoder* block_split) {
  block_split->types = (uint8_t*)BROTLI_DECODER_ALLOC(
                        s, sizeof(uint8_t) * BROTLI_INIT_STORED_BLOCK_SPLITS);
  block_split->positions_begin = (uint32_t*)BROTLI_DECODER_ALLOC(
                       s, sizeof(uint32_t) * BROTLI_INIT_STORED_BLOCK_SPLITS);
  block_split->positions_end = (uint32_t*)BROTLI_DECODER_ALLOC(
                       s, sizeof(uint32_t) * BROTLI_INIT_STORED_BLOCK_SPLITS);
  block_split->num_types = 0;
  block_split->num_types_prev_metablocks = 0;
  block_split->num_blocks = 0;
  block_split->types_alloc_size = BROTLI_INIT_STORED_BLOCK_SPLITS;
  block_split->positions_alloc_size = BROTLI_INIT_STORED_BLOCK_SPLITS;
}

BROTLI_BOOL BrotliDecoderStateInit(BrotliDecoderState* s,
    brotli_alloc_func alloc_func, brotli_free_func free_func, void* opaque) {
  if (!alloc_func) {
    s->alloc_func = BrotliDefaultAllocFunc;
    s->free_func = BrotliDefaultFreeFunc;
    s->memory_manager_opaque = 0;
  } else {
    s->alloc_func = alloc_func;
    s->free_func = free_func;
    s->memory_manager_opaque = opaque;
  }

  s->error_code = 0; /* BROTLI_DECODER_NO_ERROR */

  BrotliInitBitReader(&s->br);
  s->state = BROTLI_STATE_UNINITED;
  s->large_window = 0;
  s->substate_metablock_header = BROTLI_STATE_METABLOCK_HEADER_NONE;
  s->substate_uncompressed = BROTLI_STATE_UNCOMPRESSED_NONE;
  s->substate_decode_uint8 = BROTLI_STATE_DECODE_UINT8_NONE;
  s->substate_read_block_length = BROTLI_STATE_READ_BLOCK_LENGTH_NONE;

  s->buffer_length = 0;
  s->loop_counter = 0;
  s->pos = 0;
  s->rb_roundtrips = 0;
  s->partial_pos_out = 0;

  s->commands = NULL;
  s->commands_size = 0;
  s->commands_alloc_size = 0;
  if (s->save_info_for_recompression) {
    InitBlockSplitStored(s, &s->literals_block_splits);
    InitBlockSplitStored(s, &s->insert_copy_length_block_splits);
  }

  s->saved_position_literals_begin = BROTLI_FALSE;
  s->saved_position_lengths_begin = BROTLI_FALSE;

  s->block_type_trees = NULL;
  s->block_len_trees = NULL;
  s->ringbuffer = NULL;
  s->ringbuffer_size = 0;
  s->new_ringbuffer_size = 0;
  s->ringbuffer_mask = 0;

  s->context_map = NULL;
  s->context_modes = NULL;
  s->dist_context_map = NULL;
  s->context_map_slice = NULL;
  s->dist_context_map_slice = NULL;

  s->literal_hgroup.codes = NULL;
  s->literal_hgroup.htrees = NULL;
  s->insert_copy_hgroup.codes = NULL;
  s->insert_copy_hgroup.htrees = NULL;
  s->distance_hgroup.codes = NULL;
  s->distance_hgroup.htrees = NULL;

  s->is_last_metablock = 0;
  s->is_uncompressed = 0;
  s->is_metadata = 0;
  s->should_wrap_ringbuffer = 0;
  s->canny_ringbuffer_allocation = 1;

  s->window_bits = 0;
  s->max_distance = 0;
  s->dist_rb[0] = 16;
  s->dist_rb[1] = 15;
  s->dist_rb[2] = 11;
  s->dist_rb[3] = 4;
  s->dist_rb_idx = 0;
  s->block_type_trees = NULL;
  s->block_len_trees = NULL;

  s->mtf_upper_bound = 63;

  s->dictionary = BrotliGetDictionary();
  s->transforms = BrotliGetTransforms();

  return BROTLI_TRUE;
}

void BrotliDecoderStateMetablockBegin(BrotliDecoderState* s) {
  s->meta_block_remaining_len = 0;
  s->block_length[0] = 1U << 24;
  s->block_length[1] = 1U << 24;
  s->block_length[2] = 1U << 24;
  s->num_block_types[0] = 1;
  s->num_block_types[1] = 1;
  s->num_block_types[2] = 1;
  s->block_type_rb[0] = 1;
  s->block_type_rb[1] = 0;
  s->block_type_rb[2] = 1;
  s->block_type_rb[3] = 0;
  s->block_type_rb[4] = 1;
  s->block_type_rb[5] = 0;
  s->context_map = NULL;
  s->context_modes = NULL;
  s->dist_context_map = NULL;
  s->context_map_slice = NULL;
  s->literal_htree = NULL;
  s->dist_context_map_slice = NULL;
  s->dist_htree_index = 0;
  s->context_lookup = NULL;
  s->literal_hgroup.codes = NULL;
  s->literal_hgroup.htrees = NULL;
  s->insert_copy_hgroup.codes = NULL;
  s->insert_copy_hgroup.htrees = NULL;
  s->distance_hgroup.codes = NULL;
  s->distance_hgroup.htrees = NULL;

  /* If needed save the start of a first in metablock block */
  if (s->save_info_for_recompression) {
    BrotliEnsureCapacityBlockSplits(s, &s->literals_block_splits,
                                    s->literals_block_splits.num_blocks + 1);
    s->literals_block_splits.types[s->literals_block_splits.num_blocks] = s->literals_block_splits.num_types_prev_metablocks;
    s->literals_block_splits.positions_begin[s->literals_block_splits.num_blocks] = s->pos + (s->rb_roundtrips << s->window_bits);
    s->saved_position_literals_begin = BROTLI_TRUE;
    s->literals_block_splits.num_types =
                        BROTLI_MAX(size_t, s->literals_block_splits.num_types,
                        s->literals_block_splits.types[s->literals_block_splits.num_blocks] + 1);

    BrotliEnsureCapacityBlockSplits(s, &s->insert_copy_length_block_splits,
                                    s->insert_copy_length_block_splits.num_blocks + 1);
    s->insert_copy_length_block_splits.types[s->insert_copy_length_block_splits.num_blocks] = s->insert_copy_length_block_splits.num_types_prev_metablocks;
    s->insert_copy_length_block_splits.positions_begin[s->insert_copy_length_block_splits.num_blocks] = s->pos + (s->rb_roundtrips << s->window_bits);
    s->saved_position_lengths_begin = BROTLI_TRUE;
    s->insert_copy_length_block_splits.num_types =
                        BROTLI_MAX(size_t, s->insert_copy_length_block_splits.num_types,
                        s->insert_copy_length_block_splits.types[s->insert_copy_length_block_splits.num_blocks] + 1);
  }
}

void BrotliDecoderStateCleanupAfterMetablock(BrotliDecoderState* s) {
  BROTLI_DECODER_FREE(s, s->context_modes);
  BROTLI_DECODER_FREE(s, s->context_map);
  BROTLI_DECODER_FREE(s, s->dist_context_map);
  BROTLI_DECODER_FREE(s, s->literal_hgroup.htrees);
  BROTLI_DECODER_FREE(s, s->insert_copy_hgroup.htrees);
  BROTLI_DECODER_FREE(s, s->distance_hgroup.htrees);

  /* If needed save the end of a last in metablock block */
  if (s->save_info_for_recompression) {
    /* Save the end only if previously saved a start */
    if (s->saved_position_literals_begin) {
      s->literals_block_splits.positions_end[s->literals_block_splits.num_blocks] = s->pos + (s->rb_roundtrips << s->window_bits);
      s->literals_block_splits.num_blocks++;
      s->literals_block_splits.num_types_prev_metablocks = s->literals_block_splits.num_types;
      s->saved_position_literals_begin = BROTLI_FALSE;
    }
    if (s->saved_position_lengths_begin) {
      s->insert_copy_length_block_splits.positions_end[s->insert_copy_length_block_splits.num_blocks] = s->pos + (s->rb_roundtrips << s->window_bits);
      s->insert_copy_length_block_splits.num_blocks++;
      s->insert_copy_length_block_splits.num_types_prev_metablocks = s->insert_copy_length_block_splits.num_types;
      s->saved_position_lengths_begin = BROTLI_FALSE;
    }
  }
}

void BrotliDecoderStateCleanup(BrotliDecoderState* s) {
  BrotliDecoderStateCleanupAfterMetablock(s);

  BROTLI_DECODER_FREE(s, s->ringbuffer);
  BROTLI_DECODER_FREE(s, s->block_type_trees);
}

BROTLI_BOOL BrotliDecoderHuffmanTreeGroupInit(BrotliDecoderState* s,
    HuffmanTreeGroup* group, uint32_t alphabet_size_max,
    uint32_t alphabet_size_limit, uint32_t ntrees) {
  /* Pack two allocations into one */
  const size_t max_table_size =
      kMaxHuffmanTableSize[(alphabet_size_limit + 31) >> 5];
  const size_t code_size = sizeof(HuffmanCode) * ntrees * max_table_size;
  const size_t htree_size = sizeof(HuffmanCode*) * ntrees;
  /* Pointer alignment is, hopefully, wider than sizeof(HuffmanCode). */
  HuffmanCode** p = (HuffmanCode**)BROTLI_DECODER_ALLOC(s,
      code_size + htree_size);
  group->alphabet_size_max = (uint16_t)alphabet_size_max;
  group->alphabet_size_limit = (uint16_t)alphabet_size_limit;
  group->num_htrees = (uint16_t)ntrees;
  group->htrees = p;
  group->codes = (HuffmanCode*)(&p[ntrees]);
  return !!p;
}

// TODO: write using templates
BROTLI_BOOL BrotliEnsureCapacityBlockSplits(
    BrotliDecoderState* s, BlockSplitFromDecoder* block_splits,
    size_t requested_size) {
  if (block_splits->types_alloc_size >= requested_size &&
      block_splits->positions_alloc_size >= requested_size) {
    return BROTLI_TRUE;
  }
  if (block_splits->types_alloc_size < requested_size) {
      uint8_t* old_types = block_splits->types;
      block_splits->types = (uint8_t*)BROTLI_DECODER_ALLOC(
                            s, sizeof(uint8_t) * requested_size * 2);
      if (block_splits->types == 0) {
        /* Restore previous value. */
        block_splits->types = old_types;
        return BROTLI_FALSE;
      }
      if (!!old_types) {
        memcpy(block_splits->types, old_types,
               block_splits->num_blocks * sizeof(uint8_t));
        BROTLI_DECODER_FREE(s, old_types);
      }
      block_splits->types_alloc_size = 2 * requested_size;
  }
  if (block_splits->positions_alloc_size < requested_size) {
      /* Alloc for positions begin */
      uint32_t* old_positions = block_splits->positions_begin;
      block_splits->positions_begin = (uint32_t*)BROTLI_DECODER_ALLOC(
                            s, sizeof(uint32_t) * requested_size * 2);
      if (block_splits->positions_begin == 0) {
        block_splits->positions_begin = old_positions;
        return BROTLI_FALSE;
      }
      if (!!old_positions) {
        memcpy(block_splits->positions_begin, old_positions,
               block_splits->num_blocks * sizeof(uint32_t));
        BROTLI_DECODER_FREE(s, old_positions);
      }
      /* Alloc for positions end */
      old_positions = block_splits->positions_end;
      block_splits->positions_end = (uint32_t*)BROTLI_DECODER_ALLOC(
                            s, sizeof(uint32_t) * requested_size * 2);
      if (block_splits->positions_end == 0) {
        block_splits->positions_end = old_positions;
        return BROTLI_FALSE;
      }
      if (!!old_positions) {
        memcpy(block_splits->positions_end, old_positions,
               block_splits->num_blocks * sizeof(uint32_t));
        BROTLI_DECODER_FREE(s, old_positions);
      }
      block_splits->positions_alloc_size = 2 * requested_size;
  }
  return BROTLI_TRUE;
}

#if defined(__cplusplus) || defined(c_plusplus)
}  /* extern "C" */
#endif
