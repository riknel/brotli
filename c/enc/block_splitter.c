/* Copyright 2013 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Block split point selection utilities. */

#include "./block_splitter.h"

#include <string.h>  /* memcpy, memset */
#include <time.h>

#include "../common/platform.h"
#include "./bit_cost.h"
#include "./cluster.h"
#include "./command.h"
#include "./fast_log.h"
#include "./histogram.h"
#include "./memory.h"
#include "./quality.h"
#include "../include/brotli/encode.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// static const size_t kMaxLiteralHistograms = 100;
// // static const size_t kMaxCommandHistograms = 50;
// static const size_t kMaxCommandHistograms = 50;
// static const double kLiteralBlockSwitchCost = 28.1;
// static const double kCommandBlockSwitchCost = 13.5;
// static const double kDistanceBlockSwitchCost = 14.6;
// static const size_t kLiteralStrideLength = 70;
// // static const size_t kCommandStrideLength = 40;
// static const size_t kCommandStrideLength = 700;
// static const size_t kSymbolsPerLiteralHistogram = 544;
// // static const size_t kSymbolsPerCommandHistogram = 530;
// static const size_t kSymbolsPerCommandHistogram = 1024;
// // static const size_t kSymbolsPerDistanceHistogram = 544;
// static const size_t kSymbolsPerDistanceHistogram = 1024;
// // static const size_t kMinLengthForBlockSplitting = 128;
// static const size_t kMinLengthForBlockSplitting = 300;
// // static const size_t kIterMulForRefining = 2;
// static const size_t kIterMulForRefining = 1;
// // static const size_t kMinItersForRefining = 100;
// static const size_t kMinItersForRefining = 300;

static size_t CountLiterals(const Command* cmds, const size_t num_commands) {
  /* Count how many we have. */
  size_t total_length = 0;
  size_t i;
  for (i = 0; i < num_commands; ++i) {
    total_length += cmds[i].insert_len_;
  }
  return total_length;
}

static void CopyLiteralsToByteArray(const Command* cmds,
                                    const size_t num_commands,
                                    const uint8_t* data,
                                    const size_t offset,
                                    const size_t mask,
                                    uint8_t* literals) {
  size_t pos = 0;
  size_t from_pos = offset & mask;
  size_t i;
  for (i = 0; i < num_commands; ++i) {
    size_t insert_len = cmds[i].insert_len_;
    if (from_pos + insert_len > mask) {
      size_t head_size = mask + 1 - from_pos;
      memcpy(literals + pos, data + from_pos, head_size);
      from_pos = 0;
      pos += head_size;
      insert_len -= head_size;
    }
    if (insert_len > 0) {
      memcpy(literals + pos, data + from_pos, insert_len);
      pos += insert_len;
    }
    from_pos = (from_pos + insert_len + CommandCopyLen(&cmds[i])) & mask;
  }
}

static BROTLI_INLINE uint32_t MyRand(uint32_t* seed) {
  /* Initial seed should be 7. In this case, loop length is (1 << 29). */
  *seed *= 16807U;
  return *seed;
}

static BROTLI_INLINE double BitCost(size_t count) {
  return count == 0 ? -2.0 : FastLog2(count);
}

#define HISTOGRAMS_PER_BATCH 64
#define CLUSTERS_PER_BATCH 16

#define FN(X) X ## Literal
#define DataType uint8_t
/* NOLINTNEXTLINE(build/include) */
#include "./block_splitter_inc.h"
#undef DataType
#undef FN

#define FN(X) X ## Command
#define DataType uint16_t
/* NOLINTNEXTLINE(build/include) */
#include "./block_splitter_inc.h"
#undef FN

#define FN(X) X ## Distance
/* NOLINTNEXTLINE(build/include) */
#include "./block_splitter_inc.h"
#undef DataType
#undef FN

void BrotliInitBlockSplit(BlockSplit* self) {
  self->num_types = 0;
  self->num_blocks = 0;
  self->types = 0;
  self->lengths = 0;
  self->types_alloc_size = 0;
  self->lengths_alloc_size = 0;
}

void BrotliDestroyBlockSplit(MemoryManager* m, BlockSplit* self) {
  BROTLI_FREE(m, self->types);
  BROTLI_FREE(m, self->lengths);
}

void BrotliSplitBlock(MemoryManager* m,
                      const Command* cmds,
                      const size_t num_commands,
                      const uint8_t* data,
                      const size_t pos,
                      const size_t mask,
                      const BrotliEncoderParams* params,
                      BlockSplit* literal_split,
                      BlockSplit* insert_and_copy_split,
                      BlockSplit* dist_split,
                      BlockSplit** literals_block_splits,
                      size_t metablocks_count,
                      size_t* current_metablock) {
  {
    // printf("BrotliSplitBlock\n");
    size_t literals_count = CountLiterals(cmds, num_commands);
    uint8_t* literals = BROTLI_ALLOC(m, uint8_t, literals_count);
    if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(literals)) return;
    /* Create a continuous array of literals. */
    CopyLiteralsToByteArray(cmds, num_commands, data, pos, mask, literals);
    /* Create the block split on the array of literals.
       Literal histograms have alphabet size 256. */
    // printf("CountLiteralsBrotliSplitBlock=%zu\n", literals_count);
    if (metablocks_count == 0 || (*literals_block_splits)[*current_metablock].num_blocks == 0) {
      SplitByteVectorLiteral(
          m, literals, literals_count,
          kSymbolsPerLiteralHistogram, kMaxLiteralHistograms,
          kLiteralStrideLength, kLiteralBlockSwitchCost, params,
          literal_split);
    } else {
      // printf("STORED_BS_WAS_USED, current_metablock=%zu, metablocks_count=%zu\n", *current_metablock, metablocks_count);
      size_t stored_literals_count = 0;
      for (int i = 0; i < (*literals_block_splits)[*current_metablock].num_blocks; ++i) {
        stored_literals_count += (*literals_block_splits)[*current_metablock].lengths[i];
      }
      // TODO: think of a better approach
      if (literals_count < stored_literals_count) {
        size_t num_blocks = (*literals_block_splits)[*current_metablock].num_blocks;
        size_t to_delete = stored_literals_count - literals_count;
        while ((*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] <= to_delete) {
          printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
          (*literals_block_splits)[*current_metablock].num_blocks--;
          to_delete -= (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1];
          num_blocks--;
        }
        // if ((*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] < to_delete) {
        //   printf("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n");
        // }
        (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] -= to_delete;
      } else if (literals_count > stored_literals_count) {
        size_t num_blocks = (*literals_block_splits)[*current_metablock].num_blocks;
        (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] += (literals_count - stored_literals_count);
      }
      *literal_split = (*literals_block_splits)[*current_metablock];
      (*current_metablock)++;
      // printf("AFTER STORED_BS_WAS_USED, current_metablock=%zu, metablocks_count=%zu\n", *current_metablock, metablocks_count);
    }
    // printf("LIT BLOCK SPLIT : %zu, %zu\n Types:", literal_split->num_types, literal_split->num_blocks);
    // for (int i = 0; i < BROTLI_MIN(size_t, literal_split->num_blocks, literal_split->num_blocks); ++i) {
    //   printf("%u ", literal_split->types[i]);
    // }
    // printf("\n Length:");
    // for (int i = 0; i < BROTLI_MIN(size_t, literal_split->num_blocks, literal_split->num_blocks); ++i) {
    //   printf("%u ", literal_split->lengths[i]);
    // }
    // printf("\n");
    if (BROTLI_IS_OOM(m)) return;
    BROTLI_FREE(m, literals);
  }

  {
    /* Compute prefix codes for commands. */
    uint16_t* insert_and_copy_codes = BROTLI_ALLOC(m, uint16_t, num_commands);
    size_t i;
    if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(insert_and_copy_codes)) return;
    for (i = 0; i < num_commands; ++i) {
      insert_and_copy_codes[i] = cmds[i].cmd_prefix_;
    }
    /* Create the block split on the array of command prefixes. */
    SplitByteVectorCommand(
        m, insert_and_copy_codes, num_commands,
        kSymbolsPerCommandHistogram, kMaxCommandHistograms,
        kCommandStrideLength, kCommandBlockSwitchCost, params,
        insert_and_copy_split);
    if (BROTLI_IS_OOM(m)) return;
    /* TODO: reuse for distances? */
    BROTLI_FREE(m, insert_and_copy_codes);
  }

  {
    /* Create a continuous array of distance prefixes. */
    uint16_t* distance_prefixes = BROTLI_ALLOC(m, uint16_t, num_commands);
    size_t j = 0;
    size_t i;
    if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(distance_prefixes)) return;
    for (i = 0; i < num_commands; ++i) {
      const Command* cmd = &cmds[i];
      if (CommandCopyLen(cmd) && cmd->cmd_prefix_ >= 128) {
        distance_prefixes[j++] = cmd->dist_prefix_ & 0x3FF;
      }
    }
    /* Create the block split on the array of distance prefixes. */
    SplitByteVectorDistance(
        m, distance_prefixes, j,
        kSymbolsPerDistanceHistogram, kMaxCommandHistograms,
        kCommandStrideLength, kDistanceBlockSwitchCost, params,
        dist_split);
    if (BROTLI_IS_OOM(m)) return;
    BROTLI_FREE(m, distance_prefixes);
  }
}


// void BrotliSplitBlockFast(MemoryManager* m,
//                           const Command* cmds,
//                           const size_t num_commands,
//                           const uint8_t* data,
//                           const size_t pos,
//                           const size_t mask,
//                           const BrotliEncoderParams* params,
//                           BlockSplit* literal_split,
//                           BlockSplit* insert_and_copy_split,
//                           BlockSplit* dist_split,
//                           BlockSplit** literals_block_splits,
//                           size_t metablocks_count,
//                           size_t* current_metablock) {
//   {
//     // printf("BrotliSplitBlock\n");
//     size_t literals_count = CountLiterals(cmds, num_commands);
//     uint8_t* literals = BROTLI_ALLOC(m, uint8_t, literals_count);
//     if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(literals)) return;
//     /* Create a continuous array of literals. */
//     CopyLiteralsToByteArray(cmds, num_commands, data, pos, mask, literals);
//     /* Create the block split on the array of literals.
//        Literal histograms have alphabet size 256. */
//     // printf("CountLiteralsBrotliSplitBlock=%zu\n", literals_count);
//     if (metablocks_count == 0 || (*literals_block_splits)[*current_metablock].num_blocks == 0) {
//       SplitByteVectorLiteral(
//           m, literals, literals_count,
//           kSymbolsPerLiteralHistogram, kMaxLiteralHistograms,
//           kLiteralStrideLength, kLiteralBlockSwitchCost, params,
//           literal_split);
//     } else {
//       // printf("STORED_BS_WAS_USED, current_metablock=%zu, metablocks_count=%zu\n", *current_metablock, metablocks_count);
//       size_t stored_literals_count = 0;
//       for (int i = 0; i < (*literals_block_splits)[*current_metablock].num_blocks; ++i) {
//         stored_literals_count += (*literals_block_splits)[*current_metablock].lengths[i];
//       }
//       // TODO: think of a better approach
//       if (literals_count < stored_literals_count) {
//         size_t num_blocks = (*literals_block_splits)[*current_metablock].num_blocks;
//         size_t to_delete = stored_literals_count - literals_count;
//         while ((*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] <= to_delete) {
//           printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
//           (*literals_block_splits)[*current_metablock].num_blocks--;
//           to_delete -= (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1];
//           num_blocks--;
//         }
//         // if ((*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] < to_delete) {
//         //   printf("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n");
//         // }
//         (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] -= to_delete;
//       } else if (literals_count > stored_literals_count) {
//         size_t num_blocks = (*literals_block_splits)[*current_metablock].num_blocks;
//         (*literals_block_splits)[*current_metablock].lengths[num_blocks - 1] += (literals_count - stored_literals_count);
//       }
//       *literal_split = (*literals_block_splits)[*current_metablock];
//       (*current_metablock)++;
//       // printf("AFTER STORED_BS_WAS_USED, current_metablock=%zu, metablocks_count=%zu\n", *current_metablock, metablocks_count);
//     }
//     // printf("LIT BLOCK SPLIT : %zu, %zu\n Types:", literal_split->num_types, literal_split->num_blocks);
//     // for (int i = 0; i < BROTLI_MIN(size_t, literal_split->num_blocks, literal_split->num_blocks); ++i) {
//     //   printf("%u ", literal_split->types[i]);
//     // }
//     // printf("\n Length:");
//     // for (int i = 0; i < BROTLI_MIN(size_t, literal_split->num_blocks, literal_split->num_blocks); ++i) {
//     //   printf("%u ", literal_split->lengths[i]);
//     // }
//     // printf("\n");
//     if (BROTLI_IS_OOM(m)) return;
//     BROTLI_FREE(m, literals);
//   }
//
//   {
//     /* Compute prefix codes for commands. */
//     uint16_t* insert_and_copy_codes = BROTLI_ALLOC(m, uint16_t, num_commands);
//     size_t i;
//     if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(insert_and_copy_codes)) return;
//     for (i = 0; i < num_commands; ++i) {
//       insert_and_copy_codes[i] = cmds[i].cmd_prefix_;
//     }
//     /* Create the block split on the array of command prefixes. */
//     SplitByteVectorCommand(
//         m, insert_and_copy_codes, num_commands,
//         kSymbolsPerCommandHistogram, kMaxCommandHistograms,
//         kCommandStrideLength, kCommandBlockSwitchCost, params,
//         insert_and_copy_split);
//     if (BROTLI_IS_OOM(m)) return;
//     /* TODO: reuse for distances? */
//     BROTLI_FREE(m, insert_and_copy_codes);
//   }
//
//   {
//     /* Create a continuous array of distance prefixes. */
//     uint16_t* distance_prefixes = BROTLI_ALLOC(m, uint16_t, num_commands);
//     size_t j = 0;
//     size_t i;
//     if (BROTLI_IS_OOM(m) || BROTLI_IS_NULL(distance_prefixes)) return;
//     for (i = 0; i < num_commands; ++i) {
//       const Command* cmd = &cmds[i];
//       if (CommandCopyLen(cmd) && cmd->cmd_prefix_ >= 128) {
//         distance_prefixes[j++] = cmd->dist_prefix_ & 0x3FF;
//       }
//     }
//     /* Create the block split on the array of distance prefixes. */
//     SplitByteVectorDistance(
//         m, distance_prefixes, j,
//         kSymbolsPerDistanceHistogram, kMaxCommandHistograms,
//         kCommandStrideLength, kDistanceBlockSwitchCost, params,
//         dist_split);
//     if (BROTLI_IS_OOM(m)) return;
//     BROTLI_FREE(m, distance_prefixes);
//   }
// }


#if defined(__cplusplus) || defined(c_plusplus)
}  /* extern "C" */
#endif
