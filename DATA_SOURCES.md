# Data sources and governance

WhiteSpacer downloads source data during corpus construction and does not
redistribute it in this repository. Before each training run, verify the
current provider terms because licenses and usage policies can change.

| Source | Use in the corpus | Provider terms |
|---|---|---|
| NIST NVD | CVE descriptions, CWE mappings, CVSS, references | [NVD policies](https://nvd.nist.gov/general) |
| CISA KEV | Exploited-vulnerability prioritization and required actions | [CISA website policies](https://www.cisa.gov/about/contact-us/website-policies) |
| MITRE CWE | Weakness descriptions and defensive review prompts | [CWE terms](https://cwe.mitre.org/about/termsofuse.html) |
| MITRE CAPEC | Attack-pattern descriptions and defensive controls | [CAPEC terms](https://capec.mitre.org/about/termsofuse.html) |
| MITRE ATT&CK | Techniques, software descriptions, and detection context | [ATT&CK terms](https://attack.mitre.org/resources/terms-of-use/) |

Maintain source identifiers in every example. Do not add private incident
records, credentials, personal data, proprietary reports without permission,
or executable malware. Custom examples require documented authorization,
provenance, license compatibility, secret scanning, and representative manual
review.

The code in this repository is Apache-2.0. That license does not override the
terms of source datasets, the MIT-licensed FP base model, or the Apache-2.0
Edge0 runtime lineage.

