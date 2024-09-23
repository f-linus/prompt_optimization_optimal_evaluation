# Optimal prompt evaluation during hard-prompt tuning

Implementation of various prompt optimisation experiments.

Abstract of the corresponding paper:

> The adaptation of large language models (LLMs) to specific tasks is still relevant, despite their very general capabilities. Hard prompt tuning, i.e. the search for optimal prompts in text space, offers a compelling alternative to traditional approaches such as standard fine-tuning due to its applicability to black-box models. However, we argue that existing hard prompt tuning methods are severely limited in search step directionality due to a lack of observable objective function structure. We provide evidence for this by implementing a wide range of methods and showing that tuning parameters and design have only a limited impact on tuning performance for both a summary evaluation task and a candidate response preference task. We further hypothesise that this implies a necessary shift in focus towards search step efficiency. With search step computation cost mostly stemming from candidate prompt evaluations on a training set, we propose a novel LLM call valuation framework that allows a more optimal optimisation policy, trading off evaluation of existing prompts with the exploration of new prompts in a more information-theoretically sound manner. Finally, we show that an optimisation run using such a policy can lead to higher optimisation performance, providing evidence for the hypothesis posed.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See the various Jupyter Notebooks and further scripts in `experiments/`.
