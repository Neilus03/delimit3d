# Research direction

Updated 18 September 2026. This document supersedes the 11 September plan that
made short public-checkpoint adaptation and frozen point selection the sole
priority. The older experiments remain valid within their recorded scope.

## Direction after the supervisor meeting

The 16 September meeting called for fewer, stronger controlled experiments:
pretrain LitePT from random initialization with PartField-style supervision,
then test class-agnostic ordinary instance segmentation and interactive
segmentation. Interactive performance is complementary evidence. Both full
fine-tuning and frozen-encoder/decoder-only evaluations are acceptable, provided
each comparison holds the training mode fixed.

The causal question is whether the pretrained encoder improves a given
downstream system over that same system trained from scratch. Within each
pair, keep the decoder initialization, data, input channels, preprocessing,
schedule, seed, optimization and evaluation identical. Vary encoder
initialization only. A public pretrained checkpoint is a separate control.

The first downstream systems are:

1. LitePT + class-agnostic, mask-only Mask3D for automatic instance segmentation.
   The current implementation has no class/objectness prediction head or
   semantic cross-entropy. It uses mask matching, BCE/Dice losses, and mask
   confidence for proposal ranking.
2. LitePT + AGILE3D for interactive instance segmentation, with the exact
   upstream AGILE3D model retained as a separate external system reference.

The shared **5 cm** resolution and full **600 Mask3D / 1,100 AGILE3D epoch**
budgets were subsequent execution decisions, not a verbatim meeting matrix.
The resolution contract covers scratch training, pretraining, downstream
fine-tuning, cached geometry and evaluation. Pretraining may proceed once both
baselines have launched; their completion is not a prerequisite.

## Implemented work and unresolved controls

Scratch RGBN6 LitePT pretraining is running on Structured3D with a 336-epoch
budget. Its measured recipe and immutable inputs are external and linked in
[the experiment map](EXPERIMENTS.md). Feature-selectivity panels measure how
the representation changes; they are not downstream transfer results.

The mask-only Mask3D scratch arm and the LitePT RGB3 AGILE3D arm are running.
The latter keeps RGB-only inputs and the released ScanNet40 data/click protocol,
uses LitePT dec0 features with a single-level compatible decoder, and records
its NumPy quantization compatibility backend. It must not be described as
the unmodified released AGILE3D model.

**Open control:** RGB3 AGILE3D cannot be paired directly with an RGBN6 encoder
to isolate pretraining. Freeze a matching input/model contract and any required
weight conversion before launching the pretrained downstream arm. Likewise,
retain mask-only scoring, the same feature stages, BatchNorm policy, random
decoder and full schedule in the Mask3D pair. Do not silently reinterpret an
old custom RGBN6 run as the newer RGB3 baseline.

## What the earlier study establishes

The short-adaptation study compares frozen public LitePT against the same
checkpoint after 256 multigranular 2D-pseudomask updates. Completed ScanNet40,
S3DIS and Articulate3D evaluations support better multi-click selection under
their matched decoder recipes. A separate completed 5,000-update joint-training
control compares public versus adapted encoder initialization while training
both encoder and decoder. See [the results ledger](results/20260918_experiment_summary.md).

Feature retrieval AP, leakage, margins and visualizations support interpretation;
they do not replace automatic or interactive downstream evaluation. Object and
part metrics remain separate. Part thresholds ≥5, ≥10 and ≥20 denote minimum
surviving unambiguous tokens, with ≥10 primary.

The explanation that instance separation harms semantic grouping remains a
hypothesis. A class-aware failure does not settle the class-agnostic question.
Loss decreases, higher effective rank and colorful feature plots alone do not
demonstrate useful transfer.

## Historical failures and remaining questions

The initial ScanNet++ point-decoder study improved AP but did not pass its
fixed-IoU gate. The later 512-update run and seed repeat started from scratch
despite being intended as public-initialization controls; they cannot answer
the duration or seed-stability questions for short public adaptation. Preserve
the [seed-repeat audit](results/adaptation_256_seed20260911_native_4090.md).

PTv3 and Sonata adaptations remain secondary experiment families. Their
existence does not expand the current priority into an unrestricted backbone
sweep. The immediate evidence still needed is a completed, matched scratch
versus pretrained downstream comparison, with uncertainty and protocol limits.
