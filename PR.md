# Make a round one fair training step, and keep it fair when volunteers run different amounts of work

## Summary

CrowdGPT trains by having volunteers run a few steps on their own GPU and upload the weight
change (the "delta") they made. The coordinator merges those deltas and hands the merged model
back out. This PR fixes two ways that merge was unfair, fixes a slow start-up initialisation, and
adds a safety cap.

### The problems

1. Volunteers do not do the same amount of work. A round for a big GPU is longer than a round for a
   small one, and a volunteer may simply get through fewer steps. A plain average of deltas
   therefore weights by distance rather than by rate: whoever ran twice as many steps pulls twice
   as hard, even though their extra steps were on the same batch.
2. Volunteers are grouped into "pools" (a few volunteers that train together on the same data
   chunk, so their deltas are comparable). Pools were merged as 1/N each, regardless of how much
   data they represent, and a pool whose volunteers ran a different number of steps had no way to
   be corrected on its own.

### What changes

- Each delta is scaled by (steps it actually ran) / (steps this round asked for), so a volunteer
  contributes the right *rate* instead of the right distance. The scale is clamped to 0.5x…2.0x,
  and the reference is computed per pool: a pool that ran 63 steps is compared with 63, not with
  some global 16, which would otherwise have halved healthy big-GPU pools every round.
- Pools are merged weighted by the tokens their delta represents, and each pool carries its own
  step reference. In the normal case (everyone on target) both of these are no-ops: the scale comes
  out exactly 1.0.
- Fresh runs now start from an identity LayerNorm (weights 1, biases 0) instead of a small random
  draw. Measured effect: the first round's loss drops by roughly 1 BPB.
- A delta larger than 12000 in L2 norm is scaled down before it is applied, and every time that
  happens is logged. It has not triggered in testing; it is a tripwire for runaway submissions.
- Aggregation logs what it did: pool weights, per-delta step scale, applied / rejected / clipped
  counts, model norm before and after.

Also included (unrelated, came in with this branch's base): country geolocation for the public map
endpoints (`/api/geo`, plus admin diagnostics) and tolerance for a duplicate chunk upload racing
another one instead of returning a 500.

### The knob to turn if mainline pacing plateaus

- Nothing in this PR requires tuning, and the shipped defaults are main's: 32 micro-batches per
  optimizer step (`CROWDGPT_TARGET_ACCUM_STEPS=32`), 500 x VRAM/8 chunks per round.
- If the model plateaus early and the loss stops improving because the deltas point in inconsistent
  directions rather than because the learning rate is wrong, the batch can be shrunk:
  `CROWDGPT_TARGET_ACCUM_STEPS=2`. A smaller batch means more optimizer steps per token, and more
  steps per token measurably improves how well independent volunteers' deltas agree with each other
  (cosine agreement between two independent volunteers roughly doubles in our 4-client
  measurements).
- It is a trade, not a free win: the same tokens then take more wall-clock (about a quarter more in
  our measurements) and reach a given loss on more tokens, because the model moves less per token.
  Budget for throughput, not for a speedup.
- No client update is needed: the value travels in the ordinary task payload (`targetAccumSteps`),
  and the coordinator's step reference is derived from it per pool, so the per-delta scaling stays
  at exactly 1.0 after the change.
- Change it between rounds, not during one. Pools already training keep the value they were given;
  a round that mixes old and new values still merges correctly (each pool is compared with its own
  reference) but is not a clean comparison point for judging the loss.

### What does not change

- Wire protocol, upload frequency, upload size, batch size (32 micro-batches) and round length are
  identical to main. Uploads are still one per round per client.
- Deltas from older clients that do not report a step count are applied exactly as before, unscaled.
- No new configuration knob: the tuning knobs that existed while this was being measured were
  removed, and what ships is main's pacing plus the fixes above.

### Verification

- 46 unit tests.
- A real end-to-end run: the unmodified coordinator plus four real clients on four AMD GPUs, two
  pools of two, three rounds of 60 optimizer steps each. Every merge applied 2 of 2 deltas, rejected
  none, tripped no cap, used token weights 0.5/0.5, and produced a step scale of exactly 1.0 for
  both pools.
- Model quality on held-out data over those three rounds: cross-entropy 12.20 -> 6.40 nats/token
  (perplexity 199k -> 600). Per-round client training loss fell every round.
