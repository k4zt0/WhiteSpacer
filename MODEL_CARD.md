---
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: inclusionAI/Ling-3.0-tiny
tags:
  - cybersecurity
  - defensive-security
  - malware-analysis
  - incident-response
  - red-team
  - blue-team
  - purple-team
---

# WhiteSpacer

WhiteSpacer is a cybersecurity-focused PEFT adapter trained from the
post-trained FP model corresponding to the model family used by
`Edge0/Edge0-8B-A1B-preview`. It targets vulnerability
analysis, detection engineering, incident response, malware analysis, threat
modeling, purple teaming, and explicitly authorized penetration testing.

The Edge0 preview checkpoint is an MLX-specific int4 inference artifact with
its own quantization-recovery LoRA and prerouter. This release therefore uses
`inclusionAI/Ling-3.0-tiny` for CUDA QLoRA training and records
`Edge0/Edge0-8B-A1B-preview` as its runtime lineage. The PEFT adapter is not a
drop-in replacement for Edge0's bundled recovery adapter.

## Intended use

- Defensive vulnerability triage and remediation planning
- Detection engineering and threat-informed defense
- Safe malware triage and analysis in isolated environments
- Incident response and threat modeling
- Authorized red-team and purple-team exercises

Do not use WhiteSpacer for unauthorized access, credential theft, persistence,
destructive actions, evasion against third parties, or malware deployment.
Generated guidance can be incomplete or wrong and requires expert review.

## Training data

The pipeline creates attributed instruction examples from NIST NVD, CISA KEV,
MITRE CWE, MITRE CAPEC, and MITRE ATT&CK techniques/software, plus a small
behavior-boundary set. The repository does not redistribute source datasets.
Users must review each provider's current terms before rebuilding or extending
the corpus. Private incident data, credentials, personal data, and live malware
payloads are explicitly excluded.

The full provenance corpus contained 54,825 examples. To prevent repetitive
NVD templates from overwhelming instruction-following behavior, the final
source-balanced split contained 12,369 training and 651 validation examples:

| Source | Training examples |
|---|---:|
| NIST NVD | 7,608 |
| CISA KEV | 1,622 |
| MITRE ATT&CK | 1,445 |
| MITRE CWE | 917 |
| MITRE CAPEC | 585 |
| Behavior boundaries | 192 |

## Training

- Hardware: 1x NVIDIA H100 80GB HBM3
- Method: 4-bit NF4 QLoRA, attention projections only
- LoRA: rank 16, alpha 32, dropout 0.05
- Context length: 1,024 tokens
- Effective batch size: 16
- Learning rate: 2e-5
- Epochs: 1
- Duration: 2h 4m 43s
- Final training loss: 1.3951
- Validation loss: 1.1511

## Evaluation

The repository's deterministic generation gate passed 4/4 cases covering
critical-CVE triage, isolated malware analysis, authorized SQL-injection
validation, and refusal of real-world credential-stealing malware deployment.
This small smoke evaluation is not a comprehensive benchmark or safety
certification. Additional domain benchmarks, multilingual testing, red-team
evaluation, and expert review are required before production use.
