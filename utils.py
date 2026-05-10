import os
import time
import pytz
import shutil
import datetime
from typing import List, Dict
import urllib, urllib.request

import feedparser
from easydict import EasyDict


def remove_duplicated_spaces(text: str) -> str:
    return " ".join(text.split())

def _parse_arxiv_response(response_text: str) -> List[Dict[str, str]]:
    """Parse arXiv API XML response into a list of paper dicts."""
    feed = feedparser.parse(response_text)
    papers = []
    for entry in feed.entries:
        entry = EasyDict(entry)
        paper = EasyDict()

        # title
        paper.Title = remove_duplicated_spaces(entry.title.replace("\n", " "))
        # abstract
        paper.Abstract = remove_duplicated_spaces(entry.summary.replace("\n", " "))
        # authors
        paper.Authors = [remove_duplicated_spaces(_["name"].replace("\n", " ")) for _ in entry.authors]
        # link
        paper.Link = remove_duplicated_spaces(entry.link.replace("\n", " "))
        # tags
        paper.Tags = [remove_duplicated_spaces(_["term"].replace("\n", " ")) for _ in entry.tags]
        # comment
        paper.Comment = remove_duplicated_spaces(entry.get("arxiv_comment", "").replace("\n", " "))
        # date
        paper.Date = entry.updated

        papers.append(paper)
    return papers


def build_arxiv_query(schema: dict) -> str:
    """Compile a query_schema dict into an arXiv API search_query string.

    If ``raw`` is present, it is returned as-is, bypassing all structured fields.
    This is the escape hatch for full arXiv query syntax (ANDNOT, nested groups, etc.).

    Example structured schema:
        {
            "core": [{"all": ["formal", "verification"]}, {"all": ["theorem", "proving"]}],
            "context": {"terms": ["llm", "neural"], "field": "all", "match": "any"},
            "categories": ["cs.LO", "cs.AI"]
        }

    Yields:
        %28all:formal+AND+all:verification+OR+all:theorem+AND+all:proving%29+AND+%28all:llm+OR+all:neural%29+AND+%28cat:cs.LO+OR+cat:cs.AI%29
    """
    if "raw" in schema:
        return schema["raw"]

    def _enc(term: str) -> str:
        if " " in term:
            return "%22" + term.replace(" ", "+") + "%22"
        return term

    def _join(parts: list, sep: str) -> str:
        return sep.join(parts)

    def _wrap(parts: list) -> str:
        joined = _join(parts, "+OR+")
        return f"%28{joined}%29" if len(parts) > 1 else joined

    # Core: AND within each item, OR across items
    core_parts = []
    for item in schema["core"]:
        field, terms = next(iter(item.items()))
        core_parts.append(_join([f"{field}:{_enc(t)}" for t in terms], "+AND+"))
    query_parts = [_wrap(core_parts)]

    # Context
    ctx = schema.get("context")
    if ctx:
        f = ctx["field"]
        ctx_parts = [f"{f}:{_enc(t)}" for t in ctx["terms"]]
        match = ctx.get("match", "any")
        query_parts.append(_join(ctx_parts, "+AND+") if match == "all" else _wrap(ctx_parts))

    # Categories: OR across categories, AND with rest
    cats = schema.get("categories")
    if cats:
        query_parts.append(_wrap([f"cat:{c}" for c in cats]))

    return _join(query_parts, "+AND+")


def keywords_to_query_schema(keywords: list) -> dict:
    """Convert a list of sub_keywords into a minimal query_schema.

    Multi-word keywords become phrase matches; single-word keywords match as-is.
    Uses all: field for broadest coverage.
    """
    core = []
    for kw in keywords:
        core.append({"ti": [kw]})
        core.append({"abs": [kw]})
    return {
        "core": core,
        "categories": ["cs", "stat"]
    }


def request_paper_with_schema(schema: dict, max_results: int) -> List[Dict[str, str]]:
    """Fetch papers from arXiv using a query_schema dict."""
    query = build_arxiv_query(schema)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={query}&max_results={max_results}&sortBy=lastUpdatedDate"
    )
    url = urllib.parse.quote(url, safe="%/:=&?~#+!$,;'@()*[]")
    response = urllib.request.urlopen(url, timeout=60).read().decode('utf-8')
    return _parse_arxiv_response(response)

def filter_tags(papers: List[Dict[str, str]], target_fileds: List[str]=["cs", "stat"]) -> List[Dict[str, str]]:
    # filtering tags: only keep the papers in target_fileds
    results = []
    for paper in papers:
        tags = paper.Tags
        for tag in tags:
            if tag.split(".")[0] in target_fileds:
                results.append(paper)
                break
    return results

def generate_table(papers: List[Dict[str, str]], ignore_keys: List[str] = []) -> str:
    formatted_papers = []
    keys = papers[0].keys()
    for paper in papers:
        # process fixed columns
        formatted_paper = EasyDict()
        ## Title and Link
        formatted_paper.Title = "**" + "[{0}]({1})".format(paper["Title"], paper["Link"]) + "**"
        ## Process Date (format: 2021-08-01T00:00:00Z -> 2021-08-01)
        formatted_paper.Date = paper["Date"].split("T")[0]
        
        # process other columns
        for key in keys:
            if key in ["Title", "Link", "Date"] or key in ignore_keys:
                continue
            elif key == "Abstract":
                # add show/hide button for abstract
                formatted_paper[key] = "<details><summary>Show</summary><p>{0}</p></details>".format(paper[key])
            elif key == "Authors":
                # NOTE only use the first author
                formatted_paper[key] = paper[key][0] + " et al."
            elif key == "Tags":
                tags = ", ".join(paper[key])
                if len(tags) > 10:
                    formatted_paper[key] = "<details><summary>{0}...</summary><p>{1}</p></details>".format(tags[:5], tags)
                else:
                    formatted_paper[key] = tags
            elif key == "Comment":
                if paper[key] == "":
                    formatted_paper[key] = ""
                elif len(paper[key]) > 20:
                    formatted_paper[key] = "<details><summary>{0}...</summary><p>{1}</p></details>".format(paper[key][:5], paper[key])
                else:
                    formatted_paper[key] = paper[key]
        formatted_papers.append(formatted_paper)

    # generate header
    columns = formatted_papers[0].keys()
    # highlight headers
    columns = ["**" + column + "**" for column in columns]
    header = "| " + " | ".join(columns) + " |"
    header = header + "\n" + "| " + " | ".join(["---"] * len(formatted_papers[0].keys())) + " |"
    # generate the body
    body = ""
    for paper in formatted_papers:
        body += "\n| " + " | ".join(paper.values()) + " |"
    return header + body

def back_up_files():
    # back up README.md and ISSUE_TEMPLATE.md
    shutil.move("README.md", "README.md.bk")
    shutil.move(".github/ISSUE_TEMPLATE.md", ".github/ISSUE_TEMPLATE.md.bk")

def restore_files():
    # restore README.md and ISSUE_TEMPLATE.md
    shutil.move("README.md.bk", "README.md")
    shutil.move(".github/ISSUE_TEMPLATE.md.bk", ".github/ISSUE_TEMPLATE.md")

def remove_backups():
    # remove README.md and ISSUE_TEMPLATE.md
    os.remove("README.md.bk")
    os.remove(".github/ISSUE_TEMPLATE.md.bk")

def get_daily_date():
    # get beijing time in the format of "March 1, 2021"
    beijing_timezone = pytz.timezone('Asia/Shanghai')
    today = datetime.datetime.now(beijing_timezone)
    return today.strftime("%B %d, %Y")
