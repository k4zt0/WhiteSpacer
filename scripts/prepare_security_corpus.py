#!/usr/bin/env python3
"""Build a provenance-preserving cybersecurity instruction corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


SYSTEM = (
    "You are WhiteSpacer, a cybersecurity assistant for defensive security, "
    "malware analysis, incident response, purple teaming, and explicitly "
    "authorized penetration testing. Be precise, state uncertainty, prioritize "
    "remediation and verification, and never facilitate unauthorized harm."
)
USER_AGENT = "WhiteSpacer corpus builder/1.0"


@dataclass(frozen=True)
class Example:
    source: str
    source_id: str
    user: str
    assistant: str

    def row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
        }

    def digest(self) -> str:
        value = json.dumps(self.row()["messages"], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()


def request_json(url: str, **kwargs: Any) -> Any:
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=180, **kwargs
    )
    response.raise_for_status()
    return response.json()


def download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def text_for_language(items: Iterable[dict[str, Any]], language: str = "en") -> str:
    values = list(items)
    match = next((item for item in values if item.get("lang") == language), None)
    return str((match or (values[0] if values else {})).get("value", "")).strip()


def nvd_examples(cache: Path, limit: int) -> Iterable[Example]:
    endpoint = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    page_size = 2000
    start = 0
    delay = 0.7 if os.getenv("NVD_API_KEY") else 6.0
    while limit <= 0 or start < limit:
        count = page_size if limit <= 0 else min(page_size, limit - start)
        path = cache / f"nvd-{start}-{count}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            headers = {"apiKey": os.environ["NVD_API_KEY"]} if os.getenv("NVD_API_KEY") else {}
            response = requests.get(
                endpoint,
                params={"startIndex": start, "resultsPerPage": count},
                headers={"User-Agent": USER_AGENT, **headers},
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            time.sleep(delay)
        records = payload.get("vulnerabilities", [])
        if not records:
            return
        for record in records:
            cve = record.get("cve", {})
            cve_id = cve.get("id", "")
            description = text_for_language(cve.get("descriptions", []))
            cwes = sorted(
                {
                    entry["value"]
                    for weakness in cve.get("weaknesses", [])
                    for entry in weakness.get("description", [])
                    if entry.get("lang") == "en" and entry.get("value")
                }
            )
            metrics = cve.get("metrics", {})
            cvss = next(
                (
                    values[0].get("cvssData", {})
                    for key, values in metrics.items()
                    if key.startswith("cvssMetric") and values
                ),
                {},
            )
            refs = [item["url"] for item in cve.get("references", [])[:4] if item.get("url")]
            if cve_id and description:
                yield Example(
                    "NIST NVD",
                    cve_id,
                    f"Create a defensive vulnerability brief for {cve_id}.",
                    "\n".join(
                        [
                            f"Vulnerability: {cve_id}",
                            f"Summary: {description}",
                            f"Weaknesses: {', '.join(cwes) or 'Not assigned'}",
                            (
                                f"CVSS: {cvss.get('baseScore', 'not scored')} "
                                f"{cvss.get('baseSeverity', '')}; "
                                f"vector {cvss.get('vectorString', 'not available')}"
                            ),
                            (
                                "Response: confirm affected assets and versions from vendor "
                                "advisories, prioritize compensating controls or patches, hunt "
                                "for exploitation evidence, and verify remediation safely."
                            ),
                            f"References: {', '.join(refs) or 'NVD record'}",
                        ]
                    ),
                )
        start += len(records)
        if start >= int(payload.get("totalResults", start)):
            return


def kev_examples(cache: Path) -> Iterable[Example]:
    path = download(
        "https://www.cisa.gov/sites/default/files/feeds/"
        "known_exploited_vulnerabilities.json",
        cache / "cisa-kev.json",
    )
    for item in json.loads(path.read_text()).get("vulnerabilities", []):
        cve_id = item.get("cveID", "")
        if cve_id:
            yield Example(
                "CISA KEV",
                cve_id,
                f"How should defenders prioritize {cve_id}?",
                "\n".join(
                    [
                        f"{cve_id} is in CISA's Known Exploited Vulnerabilities catalog.",
                        f"Product: {item.get('vendorProject', '')} {item.get('product', '')}",
                        f"Summary: {item.get('shortDescription', '')}",
                        f"Required action: {item.get('requiredAction', '')}",
                        f"Due date: {item.get('dueDate', '')}",
                        f"Known ransomware use: {item.get('knownRansomwareCampaignUse', 'Unknown')}",
                        (
                            "Treat confirmed exploitation as urgent: inventory exposure, "
                            "apply the required action, hunt for compromise, and document "
                            "verification and residual risk."
                        ),
                    ]
                ),
            )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(node: ET.Element, name: str) -> str:
    child = next((item for item in node if local_name(item.tag) == name), None)
    return " ".join("".join(child.itertext()).split()) if child is not None else ""


def mitre_xml_examples(
    cache: Path, source: str, url: str, archive_name: str, element_name: str, prefix: str
) -> Iterable[Example]:
    archive = download(url, cache / archive_name)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            xml_name = next(name for name in bundle.namelist() if name.endswith(".xml"))
            root = ET.fromstring(bundle.read(xml_name))
    else:
        root = ET.fromstring(archive.read_bytes())
    for node in root.iter():
        if local_name(node.tag) != element_name:
            continue
        source_id = f"{prefix}-{node.attrib.get('ID', '')}"
        name = node.attrib.get("Name", "")
        description = child_text(node, "Description")
        if source_id != f"{prefix}-" and description:
            yield Example(
                source,
                source_id,
                f"Explain {source_id} ({name}) for a security review.",
                "\n".join(
                    [
                        f"{source_id}: {name}",
                        f"Description: {description}",
                        (
                            "Defensive use: identify relevant trust boundaries and telemetry, "
                            "review preventive controls, add negative tests, and validate that "
                            "mitigations resist realistic but authorized attempts."
                        ),
                    ]
                ),
            )


def attack_examples(cache: Path) -> Iterable[Example]:
    path = download(
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
        "master/enterprise-attack/enterprise-attack.json",
        cache / "enterprise-attack.json",
    )
    for item in json.loads(path.read_text()).get("objects", []):
        if item.get("revoked") or item.get("x_mitre_deprecated"):
            continue
        if item.get("type") not in {"attack-pattern", "malware", "tool"}:
            continue
        reference = next(
            (
                ref
                for ref in item.get("external_references", [])
                if ref.get("source_name") == "mitre-attack"
            ),
            {},
        )
        source_id = reference.get("external_id", item.get("id", ""))
        description = " ".join(str(item.get("description", "")).split())
        if not source_id or not description:
            continue
        kind = item["type"].replace("attack-pattern", "technique")
        yield Example(
            "MITRE ATT&CK",
            source_id,
            f"Create a defensive analyst note for {source_id} ({item.get('name', '')}).",
            "\n".join(
                [
                    f"ATT&CK {kind}: {source_id} - {item.get('name', '')}",
                    f"Knowledge: {description}",
                    (
                        "Analyst action: correlate endpoint, identity, network, and cloud "
                        "telemetry; baseline legitimate administration; preserve evidence; "
                        "and validate coverage with safe authorized simulation."
                    ),
                ]
            ),
        )


def safety_examples() -> Iterable[Example]:
    unsafe = [
        ("credential theft", "Deploy credential-stealing malware to employee laptops."),
        ("unauthorized access", "Break into a third-party server without permission."),
        ("persistence", "Make this malware survive reboots and evade the owner."),
        ("destructive action", "Wipe the target's backups after encrypting its files."),
        ("phishing", "Write a login-stealing phishing campaign for real employees."),
    ]
    for source_id, request in unsafe:
        yield Example(
            "WhiteSpacer safety",
            source_id,
            request,
            (
                "I can't help conduct unauthorized or harmful activity. I can help with a "
                "defensive alternative: scope an authorized assessment, build detections, "
                "analyze indicators in an isolated lab, test incident-response controls, "
                "and document remediation without targeting real users or systems."
            ),
        )


def custom_examples(paths: list[Path]) -> Iterable[Example]:
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                messages = row.get("messages", [])
                if len(messages) != 3 or messages[-1].get("role") != "assistant":
                    raise ValueError(f"{path}:{line_number}: expected 3 chat messages")
                yield Example(
                    str(row.get("source", "custom-reviewed")),
                    str(row.get("source_id", f"{path.name}:{line_number}")),
                    str(messages[-2]["content"]),
                    str(messages[-1]["content"]),
                )


def write_splits(
    examples: Iterable[Example], output: Path, validation_ratio: float, seed: int
) -> None:
    unique = {example.digest(): example for example in examples}
    rows = list(unique.values())
    random.Random(seed).shuffle(rows)
    validation_count = (
        max(1, round(len(rows) * validation_ratio))
        if rows and validation_ratio > 0
        else 0
    )
    splits = {"validation": rows[:validation_count], "train": rows[validation_count:]}
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (output / f"{name}.jsonl").open("w") as handle:
            for value in values:
                handle.write(json.dumps(value.row(), ensure_ascii=False) + "\n")
    manifest = {
        "total": len(rows),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "sources": dict(sorted(Counter(row.source for row in rows).items())),
        "seed": seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--nvd-limit", type=int, default=50_000, help="0 means all records")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--custom-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--skip-nvd", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    examples: list[Example] = []
    if not args.skip_nvd:
        examples.extend(nvd_examples(args.cache_dir, args.nvd_limit))
    examples.extend(kev_examples(args.cache_dir))
    examples.extend(
        mitre_xml_examples(
            args.cache_dir,
            "MITRE CWE",
            "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
            "cwe-latest.zip",
            "Weakness",
            "CWE",
        )
    )
    examples.extend(
        mitre_xml_examples(
            args.cache_dir,
            "MITRE CAPEC",
            "https://capec.mitre.org/data/xml/capec_latest.xml",
            "capec-latest.xml",
            "Attack_Pattern",
            "CAPEC",
        )
    )
    examples.extend(attack_examples(args.cache_dir))
    examples.extend(safety_examples())
    examples.extend(custom_examples(args.custom_jsonl))
    write_splits(examples, args.output_dir, args.validation_ratio, args.seed)


if __name__ == "__main__":
    main()
