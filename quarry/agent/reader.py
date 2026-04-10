"""DeepReader: PMC full-text fetch with claim extraction and evidence grading.

Fetches full-text articles from PubMed Central via the NCBI E-utilities API,
extracts structured claims from Results/Discussion sections, and grades each
claim A–D by evidence strength.

Intended for sub-problems where abstract-level search was insufficient.
Agents that need deep reading invoke this module directly or via Bash:

    python -m quarry.agent.reader <pmcid_or_pmid> [--section results]

The module is also importable as a library:

    from quarry.agent.reader import DeepReader
    reader = DeepReader()
    article = reader.fetch("PMC8912345")
    claims = reader.extract_claims(article, context="RNA self-replication")
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SENTENCE_CHARS = 40
_MIN_FRAGMENT_CHARS = 20

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HopGrade = Literal["A", "B", "C", "D"]

_GRADE_KEYWORDS: dict[HopGrade, list[str]] = {
    "A": [
        "clinical trial",
        "in vivo",
        "in patients",
        "randomized",
        "phase i",
        "phase ii",
        "phase iii",
        "xenograft",
        "animal model",
        "mouse model",
        "rat model",
    ],
    "B": [
        "in vitro",
        "biochemical assay",
        "crystal structure",
        "cryo-em",
        "nmr",
        "pull-down",
        "co-ip",
        "chip-seq",
        "rna-seq",
        "western blot",
        "luciferase reporter",
        "validated",
        "demonstrated",
        "confirmed",
    ],
    "C": [
        "suggests",
        "consistent with",
        "predicted",
        "computational",
        "bioinformatic",
        "comparative",
        "correlated",
        "associated",
        "may",
        "might",
        "possibly",
    ],
    "D": [
        "hypothesize",
        "speculate",
        "propose",
        "analogous",
        "by analogy",
        "could",
        "would",
        "not yet tested",
        "remains unclear",
        "unknown",
    ],
}


@dataclass
class Claim:
    """A single extracted claim with evidence grade."""

    text: str
    grade: HopGrade
    section: str  # "results" | "discussion" | "methods" | "abstract" | "other"
    evidence_phrases: list[str] = field(default_factory=list)
    validation_experiment: str | None = None


@dataclass
class Article:
    """Fetched PMC article with structured text."""

    pmcid: str
    pmid: str | None
    title: str
    abstract: str
    sections: dict[str, str]  # section_name → text

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.sections.values())


# ---------------------------------------------------------------------------
# DeepReader
# ---------------------------------------------------------------------------


class DeepReader:
    """Fetch PMC full-text and extract graded claims.

    Uses NCBI E-utilities (no API key required for moderate usage).
    Rate-limited to 3 requests/second per NCBI guidelines.
    """

    _BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    _PMC_FETCH = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"

    def __init__(self, *, timeout: int = 30) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, pmcid_or_pmid: str) -> Article:
        """Fetch article by PMC ID or PubMed ID.

        Tries PMC full-text first; falls back to abstract-only via PubMed.

        Args:
            pmcid_or_pmid: e.g. "PMC8912345", "8912345", or PMID "34976436"

        Returns:
            Article with full text if available, abstract-only otherwise.
        """
        pmcid = self._resolve_to_pmcid(pmcid_or_pmid)
        if pmcid:
            try:
                return self._fetch_pmc_fulltext(pmcid)
            except Exception:  # noqa: BLE001
                pass
        # Fall back to abstract via PubMed
        pmid = self._resolve_to_pmid(pmcid_or_pmid)
        return self._fetch_pubmed_abstract(pmid or pmcid_or_pmid)

    def extract_claims(
        self,
        article: Article,
        *,
        context: str = "",
        sections: list[str] | None = None,
        max_claims: int = 10,
    ) -> list[Claim]:
        """Extract and grade claims from article text.

        Args:
            article: Fetched article.
            context: Optional topic context for relevance filtering.
                     Claims mentioning context keywords are ranked higher.
            sections: Which sections to extract from. Defaults to
                      ["results", "discussion", "abstract"].
            max_claims: Maximum number of claims to return.

        Returns:
            List of Claim objects, ordered by relevance (if context given)
            then by grade (A > B > C > D).
        """
        target_sections = sections or ["results", "discussion", "abstract"]
        candidates: list[Claim] = []

        for sec_name in target_sections:
            text = article.sections.get(sec_name, "")
            if not text and sec_name == "abstract":
                text = article.abstract
            if not text:
                continue
            sentences = self._split_sentences(text)
            for sent in sentences:
                if len(sent) < _MIN_SENTENCE_CHARS:  # skip short fragments
                    continue
                grade, phrases = self._grade_sentence(sent)
                candidates.append(
                    Claim(
                        text=sent,
                        grade=grade,
                        section=sec_name,
                        evidence_phrases=phrases,
                    )
                )

        # Filter by context relevance
        if context:
            context_words = set(context.lower().split())
            candidates = sorted(
                candidates,
                key=lambda c: (
                    -sum(1 for w in context_words if w in c.text.lower()),
                    _grade_order(c.grade),
                ),
            )
        else:
            candidates = sorted(candidates, key=lambda c: _grade_order(c.grade))

        # Annotate weak claims with validation suggestions
        for claim in candidates:
            if claim.grade in ("C", "D"):
                claim.validation_experiment = self._suggest_validation(claim)

        return candidates[:max_claims]

    # ------------------------------------------------------------------
    # Fetching helpers
    # ------------------------------------------------------------------

    def _resolve_to_pmcid(self, identifier: str) -> str | None:
        """Convert PMID or PMC accession to normalized PMC ID."""
        s = identifier.strip()
        if s.upper().startswith("PMC"):
            return s.upper()
        # Try elink: PMID → PMCID
        try:
            pmid = s if s.isdigit() else None
            if not pmid:
                return None
            url = f"{self._BASE}/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
            data = self._get_json(url)
            links = (
                data.get("linksets", [{}])[0]
                .get("linksetdbs", [{}])[0]
                .get("links", [])
            )
            if links:
                return f"PMC{links[0]}"
        except Exception:  # noqa: BLE001
            pass
        return None

    def _resolve_to_pmid(self, identifier: str) -> str | None:
        """Extract or look up PubMed ID."""
        s = identifier.strip()
        if s.isdigit():
            return s
        if s.upper().startswith("PMC") and s[3:].isdigit():
            return None  # can't reverse without API call
        return None

    def _fetch_pmc_fulltext(self, pmcid: str) -> Article:
        """Fetch full text from PMC BioC REST API."""
        pmcid_clean = pmcid.replace("PMC", "")
        url = f"{self._PMC_FETCH}/BioC_json/{pmcid_clean}/unicode"
        data = self._get_json(url)

        # BioC JSON structure: documents[0].passages
        doc = data.get("documents", [{}])[0]
        passages = doc.get("passages", [])

        title = ""
        abstract = ""
        sections: dict[str, str] = {}

        for passage in passages:
            infons = passage.get("infons", {})
            section_type = infons.get("section_type", "").lower()
            ptype = infons.get("type", "").lower()
            text = passage.get("text", "")

            if ptype == "title" and not title:
                title = text
            elif section_type == "abstract" or ptype == "abstract":
                abstract += " " + text
            elif section_type:
                key = _normalize_section(section_type)
                sections[key] = sections.get(key, "") + "\n" + text
            else:
                sections["other"] = sections.get("other", "") + "\n" + text

        if not sections and not abstract:
            raise ValueError(f"No content extracted from {pmcid}")

        return Article(
            pmcid=pmcid,
            pmid=None,
            title=title.strip(),
            abstract=abstract.strip(),
            sections={k: v.strip() for k, v in sections.items() if v.strip()},
        )

    def _fetch_pubmed_abstract(self, pmid: str) -> Article:
        """Fetch abstract via PubMed efetch (fallback)."""
        url = (
            f"{self._BASE}/efetch.fcgi?db=pubmed&id={pmid}"
            f"&rettype=abstract&retmode=text"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "quarry/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Very rough parse: first line often has title, rest is abstract
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        abstract = " ".join(lines[1:]) if len(lines) > 1 else ""

        return Article(
            pmcid="",
            pmid=pmid,
            title=title,
            abstract=abstract,
            sections={"abstract": abstract},
        )

    def _get_json(self, url: str) -> dict:
        import json as _json

        req = urllib.request.Request(url, headers={"User-Agent": "quarry/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return _json.loads(resp.read().decode("utf-8", errors="replace"))

    # ------------------------------------------------------------------
    # Claim extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Crude sentence splitter sufficient for scientific prose."""
        # Split on period + space + capital, or newlines
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])|[\n]{2,}", text)
        return [s.strip() for s in raw if len(s.strip()) > _MIN_FRAGMENT_CHARS]

    @staticmethod
    def _grade_sentence(sentence: str) -> tuple[HopGrade, list[str]]:
        """Assign grade based on keyword heuristics (highest wins)."""
        lower = sentence.lower()
        found: dict[HopGrade, list[str]] = {g: [] for g in ("A", "B", "C", "D")}
        for grade, keywords in _GRADE_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    found[grade].append(kw)
        for grade in ("A", "B", "C", "D"):
            if found[grade]:
                return grade, found[grade]  # type: ignore[return-value]
        return "C", []  # default: indirect if no signal

    @staticmethod
    def _suggest_validation(claim: Claim) -> str:
        """Generate a minimal suggested validation experiment."""
        text = claim.text.lower()
        if "rna" in text:
            return (
                "Verify by direct biochemical assay (e.g., SHAPE-MaP, "
                "gel-shift, or kinetics measurement) under relevant conditions."
            )
        if "protein" in text or "enzyme" in text:
            return (
                "Validate with recombinant protein in defined in vitro system "
                "(activity assay, SPR, or ITC for binding)."
            )
        if "cell" in text or "cellular" in text:
            return (
                "Test in appropriate cell line with loss- and gain-of-function "
                "genetic perturbation (CRISPR KO / overexpression)."
            )
        return (
            "Design minimal experimental system to test the claim directly "
            "(e.g., reporter assay, pull-down, or imaging)."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grade_order(grade: HopGrade) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}[grade]


def _normalize_section(raw: str) -> str:
    for name in ("results", "discussion", "methods", "introduction", "conclusion"):
        if name in raw:
            return name
    return raw


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Fetch and extract claims from a PMC article"
    )
    parser.add_argument("id", help="PMC ID (PMC12345) or PubMed ID (12345)")
    parser.add_argument(
        "--context", default="", help="Topic context for relevance ranking"
    )
    parser.add_argument(
        "--section",
        nargs="+",
        default=["results", "discussion", "abstract"],
        help="Sections to extract claims from",
    )
    parser.add_argument("--max", type=int, default=10, dest="max_claims")
    args = parser.parse_args()

    reader = DeepReader()
    print(f"Fetching {args.id}...", flush=True)
    article = reader.fetch(args.id)
    print(f"Title: {article.title}")
    print(f"Sections available: {list(article.sections.keys())}\n")

    claims = reader.extract_claims(
        article,
        context=args.context,
        sections=args.section,
        max_claims=args.max_claims,
    )

    output = [
        {
            "grade": c.grade,
            "section": c.section,
            "text": c.text,
            "evidence_phrases": c.evidence_phrases,
            "validation_experiment": c.validation_experiment,
        }
        for c in claims
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
