"""Parse PubMed XML files: stream lxml iterparse → dict records.

Handles both baseline (.xml.gz) and daily update files.
Extracts: papers, authors, mesh_headings, grants, chemicals, delete_pmids.
"""

import gzip
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from lxml import etree


def _text(el: etree._Element | None) -> str:
    """Extract text content, empty string if None."""
    if el is None:
        return ""
    return (el.text or "").strip()


def _text_or_none(el: etree._Element | None) -> str | None:
    """Extract text content, None if missing or empty."""
    if el is None:
        return None
    val = (el.text or "").strip()
    return val or None


def _attr(el: etree._Element | None, name: str) -> str:
    if el is None:
        return ""
    return (el.get(name) or "").strip()


def _parse_date_elements(parent: etree._Element | None) -> date | None:
    """Parse <Year>, <Month>, <Day> child elements into date."""
    if parent is None:
        return None
    y = _text(parent.find("Year"))
    m = _text(parent.find("Month"))
    d = _text(parent.find("Day"))
    if not y:
        return None
    try:
        return date(int(y), int(m) if m else 1, int(d) if d else 1)
    except (ValueError, TypeError):
        return None


def _parse_medline_date(s: str) -> tuple[int | None, date | None]:
    """Parse MedlineDate like '2024 Jan-Feb' or '2024' → (year, date)."""
    if not s:
        return None, None
    parts = s.split()
    if parts:
        try:
            year = int(parts[0][:4])
            return year, date(year, 1, 1)
        except (ValueError, IndexError):
            pass
    return None, None


def _parse_pub_date(
    journal_issue: etree._Element | None,
) -> tuple[int | None, date | None]:
    """Extract pub_year and pub_date from JournalIssue/PubDate."""
    if journal_issue is None:
        return None, None
    pub_date_el = journal_issue.find("PubDate")
    if pub_date_el is None:
        return None, None

    # Structured date
    y = _text(pub_date_el.find("Year"))
    if y:
        m = _text(pub_date_el.find("Month"))
        d = _text(pub_date_el.find("Day"))
        year = int(y)
        # Month can be text like "Jan"
        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        try:
            mi = (
                int(m)
                if m and m.isdigit()
                else month_map.get(m[:3].lower(), 1)
                if m
                else 1
            )
            di = int(d) if d else 1
            return year, date(year, mi, di)
        except (ValueError, TypeError):
            return year, date(year, 1, 1)

    # MedlineDate fallback
    ml = _text(pub_date_el.find("MedlineDate"))
    return _parse_medline_date(ml)


def _extract_article(article: etree._Element) -> dict[str, Any]:
    """Extract paper fields from <MedlineCitation><Article>."""
    journal = article.find("Journal")
    journal_issue = journal.find("JournalIssue") if journal is not None else None

    pub_year, pub_date_val = _parse_pub_date(journal_issue)

    # Abstract: concatenate multiple AbstractText elements
    abstract_el = article.find("Abstract")
    abstract_parts = []
    if abstract_el is not None:
        for at in abstract_el.findall("AbstractText"):
            label = at.get("Label")
            text = at.text or ""
            # Include tail text from child elements (e.g., <i>, <b>)
            text += "".join(at.itertext())
            # Deduplicate: itertext() includes at.text, so just use itertext
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    abstract = " ".join(abstract_parts) or None

    # Title: can contain inline markup
    title_el = article.find("ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    title = title or None

    # Publication types
    pub_types = []
    pt_list = article.find("PublicationTypeList")
    if pt_list is not None:
        pub_types = [_text(pt) for pt in pt_list.findall("PublicationType")]

    # Language
    language = _text(article.find("Language"))

    # Pages
    pagination = article.find("Pagination")

    return {
        "title": title,
        "abstract": abstract,
        "pub_year": pub_year,
        "pub_date": pub_date_val,
        "journal_title": _text_or_none(journal.find("Title"))
        if journal is not None
        else None,
        "journal_issn": _text_or_none(journal.find("ISSN"))
        if journal is not None
        else None,
        "journal_abbr": _text_or_none(journal.find("ISOAbbreviation"))
        if journal is not None
        else None,
        "volume": _text_or_none(journal_issue.find("Volume"))
        if journal_issue is not None
        else None,
        "issue": _text_or_none(journal_issue.find("Issue"))
        if journal_issue is not None
        else None,
        "pages": _text_or_none(pagination.find("MedlinePgn"))
        if pagination is not None
        else None,
        "language": language or None,
        "pub_type": pub_types or None,
    }


def _extract_authors(pmid: int, author_list: etree._Element | None) -> list[dict]:
    """Extract author rows from <AuthorList>."""
    if author_list is None:
        return []
    rows = []
    for i, author in enumerate(author_list.findall("Author"), start=1):
        collective_el = author.find("CollectiveName")
        is_collective = collective_el is not None
        # Affiliation: take first AffiliationInfo/Affiliation
        aff_info = author.find("AffiliationInfo")
        affiliation = (
            _text_or_none(aff_info.find("Affiliation"))
            if aff_info is not None
            else None
        )

        # Collective authors store name in CollectiveName, not LastName/ForeName
        if is_collective:
            last_name = _text_or_none(collective_el)
            fore_name = None
        else:
            last_name = _text_or_none(author.find("LastName"))
            fore_name = _text_or_none(author.find("ForeName"))

        rows.append(
            {
                "pmid": pmid,
                "author_position": i,
                "last_name": last_name,
                "fore_name": fore_name,
                "initials": _text_or_none(author.find("Initials")),
                "orcid": _extract_orcid(author) or None,
                "affiliation": affiliation,
                "is_collective": is_collective,
            }
        )
    return rows


def _extract_orcid(author: etree._Element) -> str:
    """Extract ORCID from Identifier elements.

    Normalizes to XXXX-XXXX-XXXX-XXXX format:
    - URL: https://orcid.org/0000-0001-2345-6789 → bare ID
    - No hyphens: 0000000123456789 → insert hyphens
    """
    for ident in author.findall("Identifier"):
        if ident.get("Source") == "ORCID":
            val = _text(ident)
            if not val:
                continue
            # Strip URL prefix
            val = val.rsplit("/", 1)[-1]
            # Insert hyphens if bare 16-digit string
            if len(val) == 16 and val.isdigit():
                val = f"{val[:4]}-{val[4:8]}-{val[8:12]}-{val[12:]}"
            return val
    return ""


def _extract_mesh(pmid: int, mesh_list: etree._Element | None) -> list[dict]:
    """Extract mesh_headings rows. Each descriptor×qualifier = 1 row."""
    if mesh_list is None:
        return []
    rows = []
    for heading in mesh_list.findall("MeshHeading"):
        desc = heading.find("DescriptorName")
        if desc is None:
            continue
        desc_ui = _attr(desc, "UI")
        desc_name = _text(desc)
        desc_major = _attr(desc, "MajorTopicYN") == "Y"

        qualifiers = heading.findall("QualifierName")
        if qualifiers:
            for qual in qualifiers:
                rows.append(
                    {
                        "pmid": pmid,
                        "descriptor_ui": desc_ui,
                        "descriptor_name": desc_name,
                        "qualifier_ui": _attr(qual, "UI"),
                        "qualifier_name": _text(qual),
                        "is_major_topic": _attr(qual, "MajorTopicYN") == "Y",
                    }
                )
        else:
            rows.append(
                {
                    "pmid": pmid,
                    "descriptor_ui": desc_ui,
                    "descriptor_name": desc_name,
                    "qualifier_ui": None,
                    "qualifier_name": None,
                    "is_major_topic": desc_major,
                }
            )
    return rows


def _extract_grants(pmid: int, grant_list: etree._Element | None) -> list[dict]:
    """Extract grant rows from <GrantList>."""
    if grant_list is None:
        return []
    rows = []
    for grant in grant_list.findall("Grant"):
        rows.append(
            {
                "pmid": pmid,
                "grant_id": _text_or_none(grant.find("GrantID")),
                "acronym": _text_or_none(grant.find("Acronym")),
                "agency": _text_or_none(grant.find("Agency")),
                "country": _text_or_none(grant.find("Country")),
            }
        )
    return rows


def _extract_chemicals(pmid: int, chem_list: etree._Element | None) -> list[dict]:
    """Extract chemical rows from <ChemicalList>."""
    if chem_list is None:
        return []
    rows = []
    for chem in chem_list.findall("Chemical"):
        substance = chem.find("NameOfSubstance")
        rows.append(
            {
                "pmid": pmid,
                "registry_number": _text_or_none(chem.find("RegistryNumber")),
                "substance_ui": _attr(substance, "UI") or None
                if substance is not None
                else None,
                "substance_name": _text_or_none(substance),
            }
        )
    return rows


def _extract_article_ids(pubmed_data: etree._Element | None) -> dict[str, str]:
    """Extract DOI and PMC from ArticleIdList."""
    ids = {}
    if pubmed_data is None:
        return ids
    for aid in pubmed_data.findall(".//ArticleId"):
        id_type = aid.get("IdType", "")
        val = _text(aid)
        if id_type == "doi":
            ids["doi"] = val
        elif id_type == "pmc":
            ids["pmc_id"] = val
    return ids


def _extract_entrez_date(pubmed_data: etree._Element | None) -> date | None:
    """Extract created_date from History/PubMedPubDate[@PubStatus='entrez'].

    DateCreated was removed from MedlineCitation in pubmed_250101.dtd.
    The entrez date in History serves the same purpose.
    """
    if pubmed_data is None:
        return None
    history = pubmed_data.find("History")
    if history is None:
        return None
    for pd in history.findall("PubMedPubDate"):
        if pd.get("PubStatus") == "entrez":
            return _parse_date_elements(pd)
    return None


class ParseResult:
    """Accumulated parse results from one or more PubmedArticle elements."""

    __slots__ = (
        "papers",
        "authors",
        "mesh_headings",
        "grants",
        "chemicals",
        "delete_pmids",
    )

    def __init__(self):
        self.papers: list[dict] = []
        self.authors: list[dict] = []
        self.mesh_headings: list[dict] = []
        self.grants: list[dict] = []
        self.chemicals: list[dict] = []
        self.delete_pmids: list[int] = []

    def extend(self, other: "ParseResult"):
        self.papers.extend(other.papers)
        self.authors.extend(other.authors)
        self.mesh_headings.extend(other.mesh_headings)
        self.grants.extend(other.grants)
        self.chemicals.extend(other.chemicals)
        self.delete_pmids.extend(other.delete_pmids)


def parse_pubmed_article(elem: etree._Element) -> ParseResult:
    """Parse a single <PubmedArticle> element into ParseResult."""
    result = ParseResult()
    citation = elem.find("MedlineCitation")
    if citation is None:
        return result

    pmid_el = citation.find("PMID")
    if pmid_el is None or not pmid_el.text:
        return result
    pmid = int(pmid_el.text.strip())

    article = citation.find("Article")
    if article is None:
        return result

    # Paper row
    paper = _extract_article(article)
    paper["pmid"] = pmid

    # IDs from PubmedData
    pubmed_data = elem.find("PubmedData")
    article_ids = _extract_article_ids(pubmed_data)
    paper["doi"] = article_ids.get("doi")
    paper["pmc_id"] = article_ids.get("pmc_id")

    # Medline metadata
    medline_info = citation.find("MedlineJournalInfo")
    paper["country"] = (
        _text_or_none(medline_info.find("Country"))
        if medline_info is not None
        else None
    )

    # Status
    paper["medline_status"] = citation.get("Status") or None

    # Dates — DateCreated removed in pubmed_250101.dtd, use History/entrez instead
    paper["created_date"] = _extract_entrez_date(pubmed_data)
    paper["revised_date"] = _parse_date_elements(citation.find("DateRevised"))
    paper["indexed_date"] = _parse_date_elements(citation.find("DateCompleted"))

    result.papers.append(paper)

    # Child tables
    result.authors = _extract_authors(pmid, article.find("AuthorList"))
    result.mesh_headings = _extract_mesh(pmid, citation.find("MeshHeadingList"))
    result.grants = _extract_grants(pmid, article.find("GrantList"))
    result.chemicals = _extract_chemicals(pmid, citation.find("ChemicalList"))

    return result


def parse_xml_stream(path: Path) -> Iterator[ParseResult]:
    """Stream-parse a PubMed XML file, yielding ParseResult per article.

    Handles both .xml and .xml.gz files. Also yields DeleteCitation PMIDs.
    Uses iterparse for low memory usage.
    """
    open_fn = gzip.open if path.suffix == ".gz" else open

    with open_fn(path, "rb") as f:
        # Disable DTD loading/network — PubMed XML references an external DTD
        # URL that causes network hangs in lxml iterparse
        context = etree.iterparse(
            f,
            events=("end",),
            tag=("PubmedArticle", "DeleteCitation"),
            load_dtd=False,
            no_network=True,
        )

        for event, elem in context:
            if elem.tag == "PubmedArticle":
                yield parse_pubmed_article(elem)
                elem.clear()
                # Also clear parent to free memory
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

            elif elem.tag == "DeleteCitation":
                result = ParseResult()
                for pmid_el in elem.findall("PMID"):
                    if pmid_el.text:
                        result.delete_pmids.append(int(pmid_el.text.strip()))
                yield result
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]


def parse_xml_file(path: Path) -> ParseResult:
    """Parse an entire PubMed XML file into a single ParseResult."""
    combined = ParseResult()
    for result in parse_xml_stream(path):
        combined.extend(result)
    return combined
