# Compressing small transformers: a research guide for parameter golf

**The OpenAI Parameter Golf competition (launched March 18, 2026) is the exact competition matching this setup — a ~26M parameter transformer trained for 10 minutes on 8×H100, scored by bits-per-byte after INT8 quantization + zlib compression into a ≤16MB artifact.** The competition baseline scores **1.2244 BPB** and builds directly on techniques proven in the NanoGPT Speedrun. This report covers every technique axis the competition touches: weight averaging, quantization-friendly training, architecture tricks, and hyperparameter tuning, with specific paper citations and concrete numeric recommendations throughout.

---

## EMA and weight averaging: decay of 0.999 matches your warmdown window

The single most important recent paper here is **"When, Where and Why to Average Weights?"** (Ajroldi et al., February 2025, arXiv:2502.06761). It establishes that the **optimal averaging horizon is ~1% of total training budget** across diverse workloads, and that properly tuned EMA matches LAWA performance while requiring only one extra parameter copy instead of *k* copies. For 7000 steps, their 1% rule suggests a window of ~70 steps — but this was calibrated for uniform (LAWA-style) averaging, not exponential. Since EMA naturally downweights older iterates, a slightly wider effective window works better.

The foundational **LAWA paper** — "Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging" (Kaddour, 2022, arXiv:2209.14981) — found **k=10 checkpoints** optimal for RoBERTa-Base and showed that uniform averaging of the *k* latest checkpoints outperformed exponential weighting. Its follow-up, **"Early Weight Averaging meets High Learning Rates for LLM Pre-training"** (Sanyal et al., 2023, arXiv:2306.03241), adapted LAWA to nanoGPT-2 models (125M–770M params) and showed that models trained with higher learning rates benefit more from averaging — directly relevant since Muon uses aggressive LR. LAWA also mitigates loss spikes during training.

For EMA decay specifically, **"Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits"** (TMLR 2024, arXiv:2411.18704) tested decays from 0.968 to 0.998 (sampled every 16 steps) and found **α=0.998 (effective per-step ~0.99988)** dominated at peak performance. The paper confirms EMA can replace the last phase of learning rate decay and provides implicit regularization under high learning rates. For compact models, the GhostNetV3 work validated 0.9999–0.99995, with 0.99999 causing performance decline.

**The practical recommendation for 7000 steps with 1200-step warmdown is EMA decay = 0.999** (effective window ~1000 steps). This covers most of the warmdown phase, balancing smoothing against incorporating too-old iterates. At decay 0.995 the window is only ~200 steps (too responsive), while at 0.9999 the window is ~10,000 steps (exceeds total training). Running **multiple EMA decays in parallel** (0.995, 0.999, 0.9995) costs negligible memory and lets you select the best post-hoc for quantization. A LAWA alternative — saving checkpoints every ~100 steps during the last 1000 steps and uniformly averaging the last 10 — is simpler and may work equally well.

**The quantization connection is critical.** "SWALP: Stochastic Weight Averaging in Low-Precision Training" (Yang et al., 2019, arXiv:1904.11943) proved that weight averaging naturally centers solutions within flat minima, making averaged weights more compatible with quantization grids. "SQWA" (Shin et al., 2020, arXiv:2002.00343) extended this to achieve SOTA 2-bit results by placing optimized weights near the center of flat minima. The theoretical backing from **"Adam with Model EMA is Effective for Nonconvex Optimization"** (Ahn & Cutkosky, NeurIPS 2024, arXiv:2405.18199) proves clipped Adam + EMA achieves optimal convergence rates.

| EMA Decay | Effective Window | Assessment for 7000 steps |
|-----------|-----------------|---------------------------|
| 0.995 | ~200 steps | Too short; only smooths final oscillations |
| **0.999** | **~1000 steps** | **Recommended — covers warmdown, proven in practice** |
| 0.9995 | ~2000 steps | Reasonable alternative; covers warmdown + pre-warmdown |
| 0.9999 | ~10,000 steps | Too wide; overweights undertrained early checkpoints |

---

## Quantization-friendly training: kurtosis regularization beats raw weight decay

The most directly actionable paper is **"Robust Quantization: One Model to Rule Them All" (KURE)** (Shkolnik et al., NeurIPS 2020, arXiv:2002.07686). It proves that **uniformly distributed tensors are far less sensitive to quantization than Gaussian ones** (Theorem 4) and introduces kurtosis regularization targeting **kurtosis = 1.8** (the uniform distribution value, versus 3.0 for Gaussian). Standard L2/weight-decay pushes weights toward Gaussian — good but suboptimal. KURE adds `L_KURE = λ · (kurtosis(W) - 1.8)²` to the loss, directly shaping distributions for quantization.

**"R2 Loss: Range Restriction Loss for Model Compression"** (Kundu et al., Apple, ICLR 2024, arXiv:2303.08253) takes a complementary approach: three range-restricting regularizers (L∞, Margin, Soft-Min-Max) applied during pre-training remove weight outliers. MobileNet-V2 at 2-bit PTQ improved from 50.66% → 59.49% — a **>10% absolute improvement** just from training-time regularization. The **"Squashed Weight Distribution"** method (Amazon Science) reparameterizes weights as `W = tanh(P)`, mapping trained Gaussian parameters to approximately uniform distributions on [-1, 1], achieving 3-bit quantization on GLUE with only 0.2% degradation.

For weight decay specifically, higher values (WD=0.01–0.05) help by constraining weight range, but dedicated quantization regularizers are far more effective. The NanoGPT Slowrun competition (qlabs.sh/slowrun) found that **weight decay up to 16× standard practice** improves performance in data-constrained regimes — and the Parameter Golf competition is similarly data-constrained within its 10-minute window.

On **STE (Straight-Through Estimator) fake quantization**: the seminal paper is Jacob et al. (Google, CVPR 2018, arXiv:1712.05877). For INT6 specifically, QAT via STE is likely unnecessary — **FP6-LLM** (Xia et al., USENIX ATC 2024, arXiv:2401.14112) and **FlexQ** (2025, arXiv:2508.04405) both show 6-bit quantization is near-lossless (perplexity increase <0.1–0.3) for well-trained models. However, Apple's **"Compute-Optimal Quantization-Aware Training"** (Dremov et al., October 2025, arXiv:2509.22935) found that **fusing warmdown with QAT** eliminates redundant full-precision updates, since fine weight adjustments during cooldown get destroyed when QAT re-quantizes. For the Parameter Golf setup where the warmdown-then-quantize pipeline is standard, this fusion could save meaningful BPB.

**ReLU² and quantization** form a strong pairing. The **Primer** paper (So et al., NeurIPS 2021) discovered squared ReLU via NAS, and **"ReLU² Wins"** (2024, arXiv:2402.03804) confirmed it produces the highest activation sparsity (90%+) among all tested functions, with the best trade-off between sparsity and quality. Sparse activations have simpler distributions that reduce quantization error. The modded-nanogpt speedrun uses ReLU² as its default activation.

- **KURE**: Target kurtosis 1.8, add as auxiliary loss with λ~0.01 (arXiv:2002.07686)
- **R2 Loss**: Margin variant with λ=0.5 for outlier removal (arXiv:2303.08253)
- **Tanh reparameterization**: W = tanh(P) for automatically uniform weights (Amazon Science)
- **Weight decay**: WD=0.01–0.05 as baseline; up to 16× for data-constrained regimes
- **ReLU²**: Default activation choice for sparsity + quantization benefits

---

## The competition landscape: Parameter Golf is exactly this benchmark

**OpenAI Parameter Golf** (https://github.com/openai/parameter-golf, launched March 18, 2026, $1M in compute prizes) matches every specification in the query. The baseline is a **9-layer, 512-dim, 8-head (4 KV heads, GQA) transformer with 1024-token BPE vocabulary and tied embeddings** — roughly 24–26M parameters. Training runs for 10 minutes on 8×H100 SXM GPUs. The metric is **val_bpb after INT8 per-row quantization + zlib compression**, constrained to a **≤16MB artifact**. The baseline scores 1.2244 BPB at 15.86MB; an extended 4-hour non-record run by Will DePue achieved 1.2074 BPB.

The competition explicitly encourages QAT, BitNets, novel tokenizers, parameter sharing (recursive transformers, MoEUT), and test-time compute as research directions. The **Research Garden** (golf.agustif.com) tracks community hypotheses including extra RMSNorm for post-roundtrip quality, sparse outlier preservation (pQuant), and aggressive parameter sharing via Relaxed Recursive Transformers.

The direct predecessor is the **NanoGPT Speedrun** (github.com/KellerJordan/modded-nanogpt), which optimized a 124M-param model on 8×H100 for lowest wall-clock time to reach 3.28 val loss. Its current record is under 90 seconds. The speedrun framing is *L(T)* (minimize time at fixed loss), while Parameter Golf is *L(N)* (minimize loss at fixed parameter count). Both share the same optimizer (Muon), architecture innovations (RoPE, QK-Norm, ReLU², SmearGate, U-Net skips), and hardware. The **NanoGPT Slowrun** (qlabs.sh/slowrun) explores the data-efficiency axis with a fixed 100M token budget, finding that Muon + heavy regularization (16× weight decay, dropout 0.1) is essential.

Other relevant competitions include the **NeurIPS 2023 LLM Efficiency Challenge** (fine-tuning on a single GPU for 24 hours, won by QLoRA on RTX 4090) and the **Edge-Device LLM Competition** (compress models to run on 12GB smartphone RAM).

---

## Architecture: BigramHash and SmearGate are community engineering, not published papers

**BigramHash embeddings** originate from the modded-nanogpt speedrun, not from a standalone paper. The technique hashes consecutive token pairs to produce richer input representations without a full V² bigram table. The closest published work is **"Hash Layers for Large Sparse Models"** (Roller et al., 2021, arXiv:2106.04426), which discusses bigram hash routing for sparse MoE, and the foundational **"Hash Embeddings for Efficient Word Representations"** (Svenstrup et al., NeurIPS 2017, arXiv:1709.03933), which uses k hash functions selecting from a shared pool of B vectors combined with learned importance weights. With k=2 and B=4096, hash embeddings can reduce embedding parameters by 10–50× while maintaining competitive quality.

**SmearGate** is also a modded-nanogpt innovation, documented in the LessWrong post "How the NanoGPT Speedrun WR dropped by 20% in 3 months." It enables each token to blend in the previous token's representation via a tiny gated mechanism: a linear layer (12→1) produces a gating signal, and the model learns approximately `token + 0.07 × prior_token`. This replaces attention heads that would otherwise devolve into trivial "previous token" heads — freeing attention capacity for complex patterns at negligible parameter cost.

**U-Net skip connections** have strong published backing. The **Hourglass Transformer** (Nawrot et al., NAACL 2022, arXiv:2110.13711) creates a U-Net-like hierarchical structure with skip connections between mirrored encoder/decoder stages, significantly outperforming vanilla transformers on enwik8. The **Funnel-Transformer** (Dai et al., NeurIPS 2020, arXiv:2006.03236) progressively compresses sequence length through layers. In practice, modded-nanogpt uses simple U-Net skips — `x = (1-λ)·x_deep + λ·x_shallow` with learned scalar λ — connecting layers 2→11, 4→10, 6→9 for a 16-layer model. The technique also injects initial embeddings (x0) at every layer ("x0-Mixin"). **"Revisiting the Shape Convention of Transformer Language Models"** (Liao et al., February 2026, arXiv:2602.06471) found hourglass-shaped FFNs outperform conventional shapes up to 400M parameters.

For **MLP expansion ratios**, the standard SwiGLU ratio is **8/3 ≈ 2.67×** (from Shazeer's "GLU Variants Improve Transformer," 2020, arXiv:2002.05202). This emerged from parameter-matching with 4× standard FFN (3 matrices at 8/3× vs 2 matrices at 4×), not from optimization. LLaMA rounds to the nearest multiple of 256. For a 26M-param model, reducing from 4× to 2.5× may shift useful parameters from MLP to attention. The Liao et al. paper suggests this tradeoff is worth exploring at small scale.

**"Overtone initialization"** has no published paper or public documentation anywhere. This term appears to be either unpublished, community-internal, or conflated with another technique. The closest real techniques are orthogonal initialization (Saxe et al., 2013, arXiv:1312.6120) and LSUV (Mishkin & Matas, 2016), which shapes weight spectra via data-driven scaling.

---

## Hyperparameters: Muon at momentum 0.95 with trapezoidal warmdown is the proven recipe

The **Muon optimizer** was developed by Keller Jordan and collaborators (blog post December 2024, github.com/KellerJordan/Muon). It performs steepest descent under the RMS-to-RMS operator norm using Newton-Schulz iteration to approximate the matrix sign function. The theoretical foundation is in **"Old Optimizer, New Norm: An Anthology"** (Bernstein & Newhouse, 2024, arXiv:2409.20325). **"Practical Efficiency of Muon for Pretraining"** (Shah et al., Essential AI, 2025, arXiv:2505.02222) confirms Muon expands the Pareto frontier over AdamW, is more data-efficient at large batch sizes, and needs tuning over only LR and weight decay. A second study, **"Muon: Training and Trade-offs with Latent Attention and MoE"** (2025, arXiv:2509.24406), shows Muon reaches target loss with **48–52% of AdamW's training steps** for 30M–200M decoders.

| Hyperparameter | Recommended Value | Source |
|---|---|---|
| Muon momentum | **0.95**, Nesterov=True | Official repo default, validated across speedrun records |
| Newton-Schulz iterations | 5 | Sufficient with optimized quintic coefficients |
| NS coefficients | (3.4445, −4.7750, 2.0315) | Maximizes slope at zero |
| Muon LR (hidden layers) | **0.02** | modded-nanogpt; has built-in muP scaling |
| AdamW LR (embed/head) | 0.0036–0.006 | For 1D params, embeddings, classifier head |
| Warmdown ratio | **17% (1200/7000 steps)** | Validated: literature supports 10–20%, Karpathy found up to 40% |
| Warmdown shape | **Linear (trapezoidal schedule)** | Matches cosine, easier to tune; D2Z shown superior |
| Batch size | **262K–524K tokens/step** | 16 seqs/GPU × 2048 seq_len × 8 GPUs = 262K |
| Weight decay | 0.0–0.1, tune with Muon | Only two HPs need tuning: LR and WD |
| Gradient clipping | **None** | Safe to remove with Muon for speed |

The warmdown ratio of 1200/7000 (~17%) sits squarely in the validated 10–20% range. **"Training Dynamics of the Cooldown Stage in WSD LR Scheduler"** (Dremov et al., TMLR 2025, arXiv:2508.01483) analyzed cooldown specifically, recommending ~20% of total steps, sqrt-shaped or 0.7-lowered linear shapes, and **higher β₂ during cooldown**. The key insight from "Understanding Warmup-Stable-Decay Learning Rates" (Wen et al., 2024) is that during the stable phase loss remains elevated due to oscillations, while decay rapidly reveals true optimization progress — both phases are critical.

For **learning rate transfer**, **μP (Maximal Update Parameterization)** (Yang et al., 2022, arXiv:2203.03466) enables tuning on small proxy models and transferring optimal LR to full size. Muon has built-in muP scaling: `Muon LR ≈ 0.2√n × AdamW LR`. A practical caveat from **"Weight Decay may matter more than muP"** (Kosson et al., 2025, arXiv:2510.19093) is that muP assumptions hold only briefly; weight decay is what actually stabilizes LR transfer in practice.

For **batch size**, "Critical Batch Size Revisited" (2025, arXiv:2505.23971) found CBS plateaus around ~4096 sequences for language models and recommends batch size warmup — OLMo 1B achieved 43% fewer gradient steps with this approach. Crucially, the Muon paper (arXiv:2505.02222) shows Muon retains data efficiency far beyond the critical batch size, so **larger batches are safe with Muon** without the efficiency degradation seen with AdamW.

---

## Conclusion: the binding constraint is the 16MB artifact, not the 10-minute clock

The Parameter Golf competition's extended 4-hour run improved BPB by only 0.017 over the 10-minute baseline, confirming that **compression quality dominates over training time**. The highest-leverage interventions are therefore: (1) quantization-friendly weight shaping during training (KURE kurtosis regularization, R2-Loss, or tanh reparameterization), (2) weight averaging via EMA at decay 0.999 to center weights in flat minima and reduce outliers before quantization, and (3) architectural parameter efficiency (hash embeddings, recursive/shared layers, SwiGLU at 8/3×).

The Muon optimizer at momentum 0.95 with a trapezoidal schedule and 17% warmdown is the empirically validated default. Fusing warmdown with QAT (per Apple's compute-optimal QAT work) may eliminate redundant precision that gets destroyed during quantization. Running parallel EMA tracks (0.995, 0.999, 0.9995) costs almost nothing and lets you select the most quantization-friendly checkpoint post-hoc. The most underexplored axis, per the community Research Garden, is the tokenizer/vocabulary — the baseline's 1024-token vocabulary is extremely small, and vocabulary design interacts strongly with BPB scoring since the metric is tokenizer-agnostic.

### Key paper reference table

| Paper | Year | ArXiv/URL | Core Finding |
|---|---|---|---|
| LAWA (Kaddour) | 2022 | 2209.14981 | Uniform averaging of k=5–10 latest checkpoints |
| Early WA for LLMs (Sanyal et al.) | 2023 | 2306.03241 | LAWA + high LR amplifies gains for LLM pretraining |
| When/Where/Why to Average (Ajroldi et al.) | 2025 | 2502.06761 | 1% of training budget is optimal averaging window |
| SWA (Izmailov et al.) | 2018 | 1803.05407 | Weight averaging → wider optima, better generalization |
| SWALP (Yang et al.) | 2019 | 1904.11943 | SWA centers solutions in flat minima; fits quantization |
| Model Soups (Wortsman et al.) | 2022 | 2203.05482 | Averaging across hyperparameter configs improves accuracy |
| EMA Dynamics (TMLR) | 2024 | 2411.18704 | Decay 0.998 best; EMA replaces late LR decay |
| KURE (Shkolnik et al.) | 2020 | 2002.07686 | Kurtosis=1.8 (uniform) minimizes quantization sensitivity |
| R2 Loss (Kundu et al., Apple) | 2024 | 2303.08253 | Range restriction during pretraining; >10% PTQ improvement |
| Compute-Optimal QAT (Dremov et al., Apple) | 2025 | 2509.22935 | Fuse warmdown with QAT; eliminates redundant FP updates |
| FP6-LLM (Xia et al.) | 2024 | 2401.14112 | 6-bit near-lossless; best quality/cost trade-off |
| ReLU² Wins | 2024 | 2402.03804 | Highest activation sparsity among all functions |
| Primer (So et al.) | 2021 | NeurIPS 2021 | Discovered ReLU² via NAS; 4× training cost reduction |
| Hash Embeddings (Svenstrup et al.) | 2017 | 1709.03933 | k hash functions + shared pool; orders-of-magnitude compression |
| Hourglass Transformer (Nawrot et al.) | 2022 | 2110.13711 | U-Net skip connections; SOTA on enwik8 byte-level |
| SwiGLU (Shazeer) | 2020 | 2002.05202 | GLU variants outperform ReLU/GELU; 8/3 expansion ratio |
| Muon (Jordan et al.) | 2024 | kellerjordan.github.io/posts/muon/ | Spectral-norm steepest descent; 48–52% of AdamW steps |
| Practical Muon (Shah et al.) | 2025 | 2505.02222 | Muon data-efficient at large batch; only tune LR+WD |
| μP (Yang et al.) | 2022 | 2203.03466 | Zero-shot HP transfer across model widths |
| WSD Cooldown Dynamics (Dremov et al.) | 2025 | 2508.01483 | 20% cooldown optimal; higher β₂ during cooldown |
| Schedule-Free (Defazio et al.) | 2024 | 2405.15682 | Unifies scheduling and averaging; won MLCommons AlgoPerf |
| Shape Convention (Liao et al.) | 2026 | 2602.06471 | Hourglass FFNs outperform conventional at ≤400M params |