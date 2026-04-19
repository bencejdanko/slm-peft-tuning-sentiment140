# Mistral 7B v0.3 PEFT Tuning on Sentiment140

## Dataset: Sentiment140

[https://huggingface.co/datasets/bdanko/sentiment140](https://huggingface.co/datasets/bdanko/sentiment140)

* Binary Sentiment Classification (0: Negative, 4: Positive).
* Training set is 5,000 samples (shuffled and stratified).
* Test set is 1,000 samples (balanced $50/50$ distribution to ensure fair evaluation).
* Preprocessing is Mapping label `4` to `1` for standard binary cross-entropy compatibility. Removal of data leakage by ensuring no overlap between training and test indices.

## Model

Model is https://huggingface.co/mistralai/Mistral-7B-v0.3.

### VRAM & PEFT Efficiency

Full fine-tuning would require 112 GB of VRAM for a 7B model (weights + gradients + AdamW states). With Parameter-Efficient Fine-Tuning (PEFT) it's more feasbile. Mistral-7B has ~7.3 billion parameters. At bfloat16 precision (2 bytes/param), the base weights occupy ~14.6 GB.

By using LoRA or Adapters, we only train < 2% of the total parameters (roughly 50M - 150M params).

At BF, we require ~20-24 GB, an A10, or RTX 3090/4090 would work.

## Methods

### LoRA (Low-Rank Adaptation)

Injects trainable low-rank matrices into the Transformer layers (specifically the $W_q$ and $W_v$ projections).
* target `q_proj`, `v_proj`
* Update weights via $\Delta W = A \times B$, where $A$ and $B$ are low-rank.

### 2. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
Rescales attention keys, values, and FFN intermediate activations with learned vectors — no weight matrices added.
* **Architecture:** Elementwise rescaling vectors injected at `k_proj`, `v_proj`, and `down_proj`.
* **Implementation:** Using PEFT `IA3Config` — fewer trainable parameters than LoRA (~0.01% of model).

---

## Hyperparameter Tuning (Optuna)

We use Optuna to maximize the F1-Score. We will run 20 trials per method.

### Search Space & Justification

| Method | Parameter | Search Space | Justification |
| :--- | :--- | :--- | :--- |
| **LoRA** | Rank ($r$) | $\{4, 8, 16, 32\}$ | Higher $r$ captures more complexity but increases VRAM. |
| | Alpha ($\alpha$) | $\{16, 32, 64\}$ | Scaling factor for the learned weights. |
| | Learning Rate | $[1 \times 10^{-5}, 5 \times 10^{-4}]$ | Critical for convergence speed and stability. |
| **IA³**     | (schema key) Bottleneck Dim | $\{32, 64, 128\}$ | Kept for Optuna study schema compatibility; IA³ has no bottleneck dim. |
| | Learning Rate | $[5 \times 10^{-5}, 1 \times 10^{-3}]$ | IA³ rescaling vectors tolerate higher rates. |
| | Dropout | $[0.0, 0.3]$ | Attention dropout applied on base model layers. |

**Compute Budget:** Total of 40 trials. Estimated 4-6 hours on an NVIDIA L4 (GCP/Colab).

---

## Evaluation Metrics
We evaluate the classification performance using the following:
* **Accuracy:** Overall correctness.
* **Precision:** Quality of positive predictions.
* **Recall:** Ability to find all positive instances.
* **F1-Score:** Harmonic mean of Precision and Recall (**Primary Metric**).

**Target Goal:** $> 93\%$ F1-Score (Bonus Tier).

## Results Table

| Method | Best Hyperparameter Set | Accuracy | Precision | Recall | F1 | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base LLM** | N/A (Zero-Shot) | | | | | — |
| **LoRA (FT)** | $r=X, \alpha=Y, lr=Z$ | | | | | |
| **IA³ (FT)** | $lr=B$, $\text{drop}=C$ | | | | | |

## Deliverables

### Models
* `bdanko/mistral-7b-sentiment-lora`: Best LoRA adapter weights.
* `bdanko/mistral-7b-sentiment-adapter`: Best IA³ adapter weights.

### Raw Data
* `bdanko/peft-sentiment-optuna-study`: HF Dataset — all 40 Optuna trials with metrics, pushed from notebook.

## Qualitative Analysis
* **IA³ vs. LoRA:** Comparison of per-trial and final-training VRAM usage, and F1 standard deviation across 20 trials as a measure of training stability.
* **Error Analysis:** Review of 3 samples where both models misclassified sentiment (e.g., sarcasm, double negatives) and how LoRA vs. IA³ predictions differ.