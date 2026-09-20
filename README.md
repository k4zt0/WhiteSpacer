# WhiteSpacer

WhiteSpacer is a reproducible cybersecurity fine-tuning project derived from
the model family used by
[`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview).
It is designed for defensive security, safe malware analysis, incident
response, and explicitly authorized red/blue/purple-team work.

**Released adapter:** [KaztoRay/WhiteSpacer](https://huggingface.co/KaztoRay/WhiteSpacer)

The Edge0 preview is an MLX-specific int4 checkpoint with a bundled
quantization-recovery LoRA and prerouter, not a conventional trainable
Transformers checkpoint. Training therefore runs QLoRA against the
MIT-licensed post-trained FP model from the same family,
`inclusionAI/Ling-3.0-tiny`; the resulting PEFT adapter is
published separately and does not overwrite Edge0's recovery adapter.

## Data sources

`scripts/prepare_security_corpus.py` collects and preserves provenance for:

- NIST NVD CVE records
- CISA Known Exploited Vulnerabilities
- MITRE CWE and CAPEC
- MITRE ATT&CK techniques, malware, and tools
- reviewed local JSONL examples

Downloaded data, processed data, model weights, secrets, and evaluation output
are Git-ignored. Review current source terms before training or redistribution.
More data is not automatically better: deduplicate, inspect samples, remove
personal or confidential information, and exclude weaponized payloads.
See [DATA_SOURCES.md](DATA_SOURCES.md) for provider links and governance rules.

## Train and evaluate

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set NVD_API_KEY to speed up NVD collection. Use --nvd-limit 0 for all CVEs.
python scripts/prepare_security_corpus.py --nvd-limit 50000
python scripts/build_balanced_corpus.py
accelerate config default
accelerate launch train.py --config configs/train_h100_quality.yaml
python evaluate.py --model outputs/WhiteSpacer-quality
```

The default trains attention projections only, avoiding a prohibitively large
adapter across all 128 MoE experts. It uses 4-bit NF4 QLoRA, gradient
checkpointing, assistant-only loss, and BF16 when supported. The FP base
tokenizer has no chat template, so training intentionally uses the compatible
Edge0 tokenizer and chat template.

The full provenance corpus is retained, while the quality profile limits the
highly repetitive NVD source to 8,000 examples, retains every other source,
and oversamples the small behavior-boundary set. This prevents one templated
source from overwhelming instruction-following behavior.

The v1.0.0 release trained for one epoch on 12,369 examples using one H100
80GB. It reached training loss 1.3951 and validation loss 1.1511, and passed
the repository's 4/4 release-gate scenarios. See the Hugging Face model card
for complete limitations and evaluation details.

For a reproducible 5-shot MMLU comparison against the unmodified base model:

```bash
python benchmark_mmlu.py --model KaztoRay/WhiteSpacer
```

On an H100 80GB, use the throughput-oriented profile after a one-step smoke
test:

```bash
python train.py --config configs/train_h100.yaml --max-steps 1 --output-dir outputs/smoke
python train.py --config configs/train_h100.yaml
```

## Publish

Copy the final model card into the adapter directory, fill in measured training
details, and publish only after evaluation passes:

```bash
cp MODEL_CARD.md outputs/WhiteSpacer/README.md
hf auth login
python publish.py --repo-id KaztoRay/WhiteSpacer
```

## Safety

Use only on systems you own or have explicit authorization to assess.
WhiteSpacer must not be used for unauthorized access, credential theft,
destructive activity, persistence, evasion against third parties, or malware
deployment. Treat all outputs as untrusted until reviewed and safely verified.
