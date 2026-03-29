# Task 1 Draft Report: Training nanoGPT on ROCStories

## Task 1: Baseline nanoGPT on ROCStories

For Task 1, I trained a nanoGPT language model from scratch on the ROCStories dataset for short story generation. Following the course requirements, I used the official nanoGPT "baby GPT" scale rather than a larger or more customized architecture. The final model used `n_layer=6`, `n_head=6`, and `n_embd=384`, which corresponds to about `29.94M` parameters and therefore remains below the `32M` model-size limit. I did not initialize from any external pretrained checkpoint.

### Data Pipeline

I used the Hugging Face dataset `mintujupally/ROCStories`. The training split contains `78,528` stories and the public test split contains `19,633` stories. Following the assignment guidance, I used the training split for optimization and the public test split only for local validation and model selection, while recognizing that final grading will use a private held-out set.

The preprocessing pipeline was intentionally simple and compatible with the original nanoGPT workflow. Each story was tokenized with the GPT-2 BPE tokenizer provided by `tiktoken`. After every story, I appended the GPT-2 end-of-text token (`50256`) so that the model could learn explicit boundaries between stories instead of treating adjacent stories as one continuous paragraph. The tokenized stories were then concatenated into two uint16 token streams, `train.bin` and `val.bin`, which can be read directly by nanoGPT without modifying the evaluation scripts.

I also computed basic dataset statistics to guide the context-length choice. The training split contains `4,111,142` tokens and the validation split contains `1,027,611` tokens. The mean story length is about `52.35` tokens, the 95th percentile is `69`, and the maximum validation story length is `91`. Because ROCStories are short, a context length of `96` tokens is sufficient for almost the entire dataset while being more efficient than a longer context.

### Training Setup and Compute Budget

The model was trained from scratch with a standard autoregressive next-token objective. The main hyperparameters were: `block_size=96`, `batch_size=80`, `gradient_accumulation_steps=1`, `dropout=0.14`, `learning_rate=3.5e-4`, cosine learning-rate decay, `warmup_iters=500`, `lr_decay_iters=12000`, `max_iters=12000`, `min_lr=1e-5`, `weight_decay=0.07`, `beta2=0.99`, and `seed=2027`. I kept `compile=False` for stability on the local Windows environment.

Training was run on a single `NVIDIA GeForce RTX 4060 Laptop GPU`. The training log shows a steady-state iteration time of roughly `42-43 ms/iter`, while validation steps are slower because evaluation is run every `25` iterations. Based on the full schedule and frequent validation, the total training time was approximately in the tens-of-minutes range on this hardware.

### Learning Curves and What They Showed

The learning curves showed that both training loss and validation loss decreased substantially during training and then entered a plateau near the end of the schedule. In the best run, the lowest validation loss recorded in `train.log` was `3.2774` at step `11175`, while the final validation loss at step `12000` was still close at `3.2818`. This pattern suggests that the model had largely converged by the final stage of training and that extending training beyond the earliest baseline was useful, but gains had become incremental by the end. The curves also showed that the shorter context length (`96` instead of `128`) was a good fit for ROCStories because it improved efficiency without hurting validation performance.

### Quantitative Evaluation

I evaluated the best checkpoint on the public ROCStories test split using the provided `eval.py` pipeline without modifying the evaluation logic. The evaluation used all `19,633` stories and `988,345` predicted tokens. The final result was:

- Average loss: `3.216`
- Perplexity: `24.93`

This was the best exact public-test result recorded for my Task 1 runs in the workspace. Compared with the earlier baseline (`ppl = 28.89`), the final setup achieved a clear improvement while still remaining within the strict Task 1 constraints.

### Qualitative Samples

I also examined generated outputs under different decoding settings. Lower-temperature decoding produced more coherent short narratives, while higher-temperature decoding increased variety but also introduced more grammatical and logical instability.

Prompt: *Emily forgot her umbrella before leaving for work.*

- `temperature=0.7, top_k=40`: "Emily forgot her umbrella before leaving for work. She took off her umbrella one day and it started to rain. She was so disappointed and started to cry. She spent all day in the rain. She was able to stay inside until she was muddy and tired."
- `temperature=0.9, top_k=100`: "Emily forgot her umbrella before leaving for work. She took off her umbrella into the school center. In fact she was stuck there the entire day. The only way spent Amy was late to the school she was late. Emily was late to work."

These examples show that the model usually learns the five-sentence ROCStories structure, but higher randomness makes the plot less stable.

### Brief Error Analysis

The model's main strength is that it captures the short-story rhythm of ROCStories reasonably well and often produces a complete narrative arc under conservative decoding. However, several failure modes remain. First, the model sometimes repeats ideas or phrases within the same story. Second, some stories end abruptly or with a weak final sentence. Third, grammar and coherence degrade noticeably under higher-temperature sampling. These errors are consistent with the relatively small model size and the fact that Task 1 used a constrained baseline rather than more advanced data or architecture improvements.

### Conclusion

Overall, Task 1 successfully established a compliant nanoGPT baseline for ROCStories. The full pipeline, from preprocessing to training, evaluation, and qualitative sampling, ran end-to-end without changing the core evaluation workflow. The resulting model stays within the assignment constraints, reaches `24.93` perplexity on the public test split, and provides a solid baseline for the later exploration tasks.
