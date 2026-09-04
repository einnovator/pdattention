# SWE-bench Verified: FIM-14B baseline card

## Published target

- Model: `TIGER-Lab/FIM-14B`
- Base: `Qwen/Qwen2.5-Coder-14B-Instruct`
- Reported score: `29.20%`, mean over three seeds
- Harness: upstream R2E-Gym edit agent, no function calling
- Dataset: `R2E-Gym/SWE-Bench-Verified`, test split, all 500 instances
- Engine: vLLM OpenAI-compatible server
- Context: 65,536 tokens; prefix caching enabled
- Decoding: temperature `0`
- Agent budget: 40 normal steps, 100 absolute steps
- Grading: official SWE-bench container harness
- Model-card revision: `c06455c0e18ae4991d5699a83f97b4edbfb21147`
- Source: <https://huggingface.co/TIGER-Lab/FIM-14B>

The exact published cell is the primary target because it is a capable 14B
coding agent with a nonzero, fully documented SWE-bench Verified result. A
quantized MLX/llama.cpp run on Apple Silicon is a useful calibration but is not
an exact reproduction of this vLLM/BF16 cell.

## Required order

1. Run two instances only to qualify model loading, the unmodified R2E-Gym
   scaffold, Docker isolation, telemetry, and official grading.
2. Run all 500 no-PRA instances with the pinned model and configuration.
3. Review statistical compatibility with `29.20%` and record every difference.
4. Run gateway pass-through only after `BASELINE_REPRODUCED`.
5. Run PRA budget conditions only after pass-through qualification.
