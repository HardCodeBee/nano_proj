# Report-Ready Qualitative Examples

This note collects concrete story samples for the prompt:

`Emily forgot her umbrella before leaving for work.`

It is meant to support two report sections:

1. `Test performance` with a concrete qualitative example
2. `Main failure modes` with specific text evidence instead of only abstract claims

## 1. Historical Task 1 examples already used in the archived report

These are the exact examples that were already written into the Task 1 report draft and summary:

- Source: `archive/task1/notes/task1_report_draft.md`
- Backup source: `archive/task1/notes/task1_summary.md`

### `temperature = 0.7, top_k = 40`

`Emily forgot her umbrella before leaving for work. She took off her umbrella one day and it started to rain. She was so disappointed and started to cry. She spent all day in the rain. She was able to stay inside until she was muddy and tired.`

How to use it in the report:

- This is the stronger of the two archived decoding examples.
- It does preserve a five-sentence ROCStories shape and keeps rain / work as the central topic.
- But it is still not fully coherent: `forgot her umbrella` becomes `took off her umbrella`, and the ending `stay inside until she was muddy and tired` is semantically awkward.

### `temperature = 0.9, top_k = 100`

`Emily forgot her umbrella before leaving for work. She took off her umbrella into the school center. In fact she was stuck there the entire day. The only way spent Amy was late to the school she was late. Emily was late to work.`

How to use it in the report:

- This is the clearest historical example for the claim that higher-temperature decoding increases instability.
- The story keeps the umbrella/work topic only loosely.
- Grammar degrades badly in the fourth sentence, there is an entity slip from `Emily` to `Amy`, and the event chain becomes hard to interpret.

Important provenance note:

- The current repository still contains these two examples in archived notes.
- The archived raw sample file `archive/task1/run_history/out-rocstories/samples.jsonl` contains a different surviving `temperature = 0.7, top_k = 40` output, so the report text and the raw artifact should not be treated as byte-identical copies of the same generation.
- The current repository does **not** appear to keep a raw current-run `jsonl` artifact for `temperature = 0.9, top_k = 100`.
- So for `0.9 / 100`, the archived Task 1 notes are the best surviving source.

## 2. Current repository examples that are useful for failure-mode analysis

These examples come from the active `out-task2/samples/` artifacts and are useful if the report wants more concrete evidence than the old Task 1 pair alone.

### A. Baseline-style low-temperature example still shows shallow coherence

- Source: `out-task2/samples/r19-baseline-openai.jsonl`
- Params: `temperature = 0.7, top_k = 40`
- Local judge: `2`

`Emily forgot her umbrella before leaving for work. She decided to sit down to watch the movie. She was so distracted that it started raining. Emily decided to take a break from work. Emily took the bus home.`

Why it is useful:

- This is a good concrete example for saying that lower-temperature decoding can look superficially orderly while still failing to produce a convincing causal chain.
- The prompt mentions leaving for work in the rain, but the continuation drifts into `watch the movie` and `take a break from work` without a natural bridge.

### B. Current E4 low-temperature sweep example: topic retained, logic still weak

- Source: `out-task2/samples/e4-posteot-mask-openai-t070-k40.jsonl`
- Params: `temperature = 0.7, top_k = 40`
- Local judge: `2`

`Emily forgot her umbrella before leaving for work. She decided to use her umbrella when she got outside. She put a towel in her backpack. She then got a sunburn. She was able to get to work on time.`

Why it is useful:

- This is a strong current example for `test performance` if you want a newer artifact rather than only the archived Task 1 sample.
- It stays short, finishes cleanly, and remains near the prompt topic.
- But it also shows why the mean judge stays low: it immediately contradicts the opening by saying she `decided to use her umbrella`, then jumps to `sunburn`, so the story is complete in form but weak in causal coherence.

### C. Logic-jump failure mode

- Source: `out-task2/samples/e8-prefix-continuation-openai-t070-k40.jsonl`
- Params: `temperature = 0.7, top_k = 40`
- Local judge: `2`

`Emily forgot her umbrella before leaving for work. She decided to use her umbrella when she got outside. While walking to the park, she heard a loud crack. She went to investigate and found a bird had dropped. She had to throw it away.`

Why it is useful:

- This is a very concrete example for the report sentence `the model struggles to sustain a coherent event chain through the middle of the story`.
- The opening is about commuting to work in bad weather, but the middle abruptly diverts to a park and a fallen bird.

### D. Severe object/logic corruption failure mode

- Source: `out-task2/samples/e4-posteot-mask-openai-t070-k120.jsonl`
- Params: `temperature = 0.7, top_k = 120`
- Local judge: `1`

`Emily forgot her umbrella before leaving for work. She wanted to sit down to get it but didn't have any. When it got cold, she put the umbrella in the washer. She took the umbrella to the store. After she got home, she drank it.`

Why it is useful:

- This is the strongest current example for explicit `main failure modes`.
- It shows object misuse and hard semantic corruption: the umbrella is washed, carried to the store, and finally `drank`.

### E. Higher-scoring sample that is still not truly robust

- Source: `out-task2/samples/e5-synth-continuation-weighted-openai.jsonl`
- Params: `temperature = 0.8, top_k = 200`
- Local judge: `3`

`Emily forgot her umbrella before leaving for work. She wanted to sit down and play in the park, so she called her mom and dad. Once they arrived at the park, they were greeted by an old lady who was having an argument about her umbrella. Relieved, they quickly put the umbrella on and continued playing in the park, enjoying their time together. When they returned home, they enjoyed a sunny afternoon and enjoyed a wonderful day together.`

Why it is useful:

- This is a helpful example for explaining why a slightly better local judge does not automatically mean the story problem is solved.
- Surface fluency is better than the harsher failure cases.
- But the story still drifts away from `leaving for work`, replaces the work/rain scenario with a family park outing, and ends with generic repetition (`enjoyed` repeated).

### F. `storymix_v1` Stage C example: better shell, still not a clean win

- Source: remote `storymix-v1-stageC-openai` sample shared during the first pilot
- Params: `temperature = 0.8, top_k = 200`
- Local judge: `3`

`Anna planted tomato seeds in her backyard. She watered them every day. Soon they sprouted! Anna picked fresh seeds! She decided to give her harvest in a garden next year!`

Why it is useful:

- This is a good example for explaining the strongest current positive signal from the new from-scratch `storymix_v1` line.
- The story is short, complete, and more coherent than many of the harsher failure cases.
- But it still shows why the line is not a clean winner: `picked fresh seeds` is semantically off, the middle remains shallow, and the improved local judge came with a much worse `ppl` than the older E4 anchor.

## 3. Suggested report wording

### For `Test performance`

Use the archived Task 1 pair if the report is specifically discussing the original baseline decoding comparison:

- `temperature = 0.7, top_k = 40` gives the more stable five-sentence story shape
- `temperature = 0.9, top_k = 100` gives higher variation but clearly worse coherence and grammar

If the report instead wants the best current repository artifact at low-temperature decoding, cite the E4 example from `e4-posteot-mask-openai-t070-k40.jsonl` and then note that it is better-formed than many alternatives but still only judge `2`.

### For `Main failure modes`

The most concrete mapping is:

- awkward coherence even under conservative decoding:
  `r19-baseline-openai.jsonl`
- contradiction and shallow causal chain:
  `e4-posteot-mask-openai-t070-k40.jsonl`
- abrupt logic jump / topic derailment:
  `e8-prefix-continuation-openai-t070-k40.jsonl`
- severe semantic corruption:
  `e4-posteot-mask-openai-t070-k120.jsonl`
- better surface fluency without true prompt-conditioned robustness:
  `e5-synth-continuation-weighted-openai.jsonl`
- cleaner short-story shell from the new from-scratch route without a true overall win:
  `storymix-v1-stageC-openai` shared sample
