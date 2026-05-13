"""
DailyArXiv filter with incremental SQLite storage.
Three stages: fetch → screen → summarize (full LaTeX reading via arxiv-to-prompt)
"""
import os
import sys
import json
import time
import sqlite3
from typing import List, Dict
from pathlib import Path
from openai import OpenAI
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    request_paper_with_schema,
    keywords_to_query_schema,
    filter_tags,
)

from datetime import datetime
import pytz

BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "papers.db")
KEYWORDS_PATH = BASE_DIR / "keywords.json"
PROMPTS_DIR = BASE_DIR / "prompts"
SUMMARY_MODEL = "deepseek-v4-pro"
SUMMARY_PARALLEL = 5

# ── Config ──────────────────────────────────────────────────────────────

def load_config():
    with open(KEYWORDS_PATH) as f:
        cfg = json.load(f)
    return cfg["sections"], cfg["defaults"]

KEYWORDS, CONFIG_DEFAULTS = load_config()

BATCH_SIZE = 5
SCREEN_PARALLEL = 8
SCREEN_MODEL = "deepseek-v4-flash"
MAX_RETRIES = 3

# ── Prompt Templates ────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    with open(PROMPTS_DIR / f"{name}.txt") as f:
        return f.read()

SCREEN_TEMPLATE = load_prompt("screen")
SUMMARY_TEMPLATE = load_prompt("summary")

# ── Database ────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            tags TEXT,
            comment TEXT,
            link TEXT,
            date TEXT,
            section TEXT,
            keep INTEGER DEFAULT NULL,
            screen_date TEXT DEFAULT NULL,
            summary TEXT DEFAULT NULL
        )
    """)
    # Migration: add columns if they don't exist (for existing DBs)
    for col, default in [("keep", "NULL"), ("screen_date", "NULL")]:
        try:
            c.execute(f"ALTER TABLE papers ADD COLUMN {col} {default}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn

def get_known_ids(conn, section: str) -> set:
    c = conn.cursor()
    c.execute("SELECT arxiv_id FROM papers WHERE section = ?", (section,))
    return {row[0] for row in c.fetchall()}

def save_paper(conn, paper: Dict, section: str):
    c = conn.cursor()
    tags = json.dumps(paper.get("Tags", []))
    c.execute("""
        INSERT OR REPLACE INTO papers
        (arxiv_id, title, abstract, tags, comment, link, date, section)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        paper.get("arxiv_id", ""),
        paper["Title"],
        paper["Abstract"],
        tags,
        paper.get("Comment", ""),
        paper["Link"],
        paper["Date"],
        section,
    ))

def get_unscreened_papers(conn, section: str) -> List[Dict]:
    c = conn.cursor()
    c.execute("""
        SELECT arxiv_id, title, abstract, tags, comment, link, date
        FROM papers
        WHERE section = ? AND keep IS NULL
        ORDER BY date DESC
    """, (section,))
    papers = []
    for row in c.fetchall():
        papers.append({
            "arxiv_id": row[0], "Title": row[1], "Abstract": row[2],
            "Tags": json.loads(row[3]) if row[3] else [],
            "Comment": row[4], "Link": row[5], "Date": row[6],
        })
    return papers

def get_unsummarized_papers(conn) -> List[Dict]:
    """Get kept papers that haven't been summarized or have truncated summaries."""
    c = conn.cursor()
    c.execute("""
        SELECT arxiv_id, title, section
        FROM papers
        WHERE keep = 1 AND summary IS NULL
        ORDER BY date DESC
    """, ())
    return [{"arxiv_id": r[0], "Title": r[1], "Section": r[2]} for r in c.fetchall()]

def get_section_papers(conn, section: str, limit: int) -> List[Dict]:
    c = conn.cursor()
    c.execute("""
        SELECT arxiv_id, title, summary, link, date
        FROM papers
        WHERE section = ? AND keep = 1
        ORDER BY date DESC
        LIMIT ?
    """, (section, limit))
    return [
        {
            "arxiv_id": r[0], "Title": r[1], "Summary": r[2],
            "Link": r[3], "Date": r[4],
        }
        for r in c.fetchall()
    ]

def update_keeps(conn, section: str, keeps: List[tuple]):
    c = conn.cursor()
    c.executemany(
        "UPDATE papers SET keep = ?, screen_date = ? WHERE arxiv_id = ?",
        [(int(k), d, a) for a, k, d in keeps],
    )
    conn.commit()

def update_summaries(conn, results: List[tuple]):
    c = conn.cursor()
    c.executemany(
        "UPDATE papers SET summary = ? WHERE arxiv_id = ?",
        [(summary, arxiv_id) for arxiv_id, summary in results],
    )
    conn.commit()

def extract_arxiv_id(link: str) -> str:
    for part in link.split("/"):
        if "." in part and part[0].isdigit():
            return part.split("v")[0]
    return link

# ── Fetch ───────────────────────────────────────────────────────────────

def fetch_papers():
    conn = init_db()
    total_fetched = 0

    for section, config in KEYWORDS.items():
        print(f"\n{'='*60}\nSection: {section}")
        known_ids = get_known_ids(conn, section)
        print(f"  Already in DB: {len(known_ids)} papers")

        all_papers = []
        max_kw = config.get("max_results_per_query", CONFIG_DEFAULTS["max_results_per_query"])

        schema = config.get("query_schema") or keywords_to_query_schema(config["sub_keywords"])
        if "raw" in schema:
            print(f"  Query schema: raw query ({len(schema['raw'])} chars), categories={schema.get('categories', [])}")
        else:
            print(f"  Query schema: {len(schema.get('core', []))} core groups, categories={schema.get('categories', [])}")

        for attempt in range(4):
            try:
                papers = request_paper_with_schema(schema, max_kw)
                papers = filter_tags(papers)
                all_papers.extend(papers)
                print(f"  Query returned {len(papers)} papers")
                break
            except Exception as e:
                if attempt < 3:
                    print(f"  arXiv API failed (attempt {attempt+1}), retrying in 30s")
                    time.sleep(30)
                else:
                    print(f"  Failed to fetch after 4 attempts")

        seen = set()
        unique_papers = []
        for p in all_papers:
            aid = extract_arxiv_id(p["Link"])
            if aid not in seen:
                seen.add(aid)
                p["arxiv_id"] = aid
                unique_papers.append(p)

        new_papers = [p for p in unique_papers if p["arxiv_id"] not in known_ids]
        print(f"  Fetched: {len(unique_papers)} unique, {len(new_papers)} new")

        for paper in new_papers:
            save_paper(conn, paper, section)
            total_fetched += 1
        conn.commit()
        time.sleep(1)

    conn.close()
    print(f"\nFetch done! {total_fetched} new papers saved to DB")

# ── Screen ──────────────────────────────────────────────────────────────

def build_screen_prompt(keyword: str, description: str, papers: List[Dict]) -> str:
    entries = []
    for i, p in enumerate(papers):
        entries.append({
            "id": i, "title": p["Title"], "abstract": p["Abstract"],
            "tags": p.get("Tags", []), "comment": p.get("Comment", ""),
        })
    papers_json = json.dumps(entries, ensure_ascii=False, indent=2)
    return SCREEN_TEMPLATE.format(keyword=keyword, description=description, papers_json=papers_json)


def screen_batch(args):
    section, description, keyword, batch = args
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
    prompt = build_screen_prompt(keyword, description, batch)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=SCREEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            if "```" in content:
                for part in content.split("```"):
                    p = part.strip()
                    if p.startswith("json"): p = p[4:].strip()
                    try:
                        result = json.loads(p)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not parse JSON: {content[:200]}")
            else:
                result = json.loads(content)

            keep_map = {item["id"]: (1 if item.get("keep") else 0) for item in result}
            return [(p["arxiv_id"], keep_map.get(i, 0)) for i, p in enumerate(batch)]

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                print(f"    Batch failed: {e}")
                return [(p["arxiv_id"], 0) for p in batch]


def screen_papers():
    conn = init_db()
    beijing_tz = pytz.timezone("Asia/Shanghai")
    current_date = datetime.now(beijing_tz).strftime("%Y-%m-%d")
    total_screened = 0

    for section, config in KEYWORDS.items():
        description = config["description"]
        print(f"\n{'='*60}\nSection: {section}")

        unscreened = get_unscreened_papers(conn, section)
        if not unscreened:
            print("  No new papers to screen")
            continue
        print(f"  Unscreened: {len(unscreened)} papers")

        batches = []
        for i in range(0, len(unscreened), BATCH_SIZE):
            batch = unscreened[i:i + BATCH_SIZE]
            batches.append((section, description, section, batch))

        with Pool(processes=SCREEN_PARALLEL) as pool:
            results = pool.map(screen_batch, batches)

        all_results = []
        for batch_results in results:
            for arxiv_id, keep in batch_results:
                all_results.append((arxiv_id, keep, current_date))

        update_keeps(conn, section, all_results)
        kept = sum(1 for _, k, _ in all_results if k)
        total_screened += len(all_results)
        print(f"  Screened {len(all_results)} papers, {kept} kept")
        time.sleep(1)

    conn.close()
    print(f"\nScreening done! {total_screened} papers screened")

# ── Summarize ───────────────────────────────────────────────────────────

def build_summary_prompt(title: str, section: str, latex_text: str) -> str:
    # Truncate to fit in context window if needed
    if len(latex_text) > 80000:
        latex_text = latex_text[:40000] + "\n\n... [truncated] ...\n\n" + latex_text[-40000:]
    return SUMMARY_TEMPLATE.format(title=title, section=section, latex_text=latex_text)


def summarize_single(paper_info: Dict) -> tuple:
    """Summarize one paper. Returns (arxiv_id, summary) or None."""
    try:
        from arxiv_to_prompt import process_latex_source
    except ImportError:
        print(f"  arxiv_to_prompt not installed, skipping summary")
        return None

    arxiv_id = paper_info["arxiv_id"]
    title = paper_info["Title"]
    section = paper_info["Section"]

    print(f"  [{arxiv_id}] Downloading LaTeX...")
    try:
        latex_text = process_latex_source(arxiv_id)
    except Exception as e:
        print(f"  Failed to download LaTeX: {e}")
        return None

    if latex_text is None:
        print(f"  No LaTeX source available for {arxiv_id}")
        return None

    prompt = build_summary_prompt(title, section, latex_text)
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()

            # Detect truncation and retry if needed
            if getattr(response.choices[0], "finish_reason", None) == "length":
                if attempt < MAX_RETRIES - 1:
                    print(f"  Response truncated, retry {attempt+1}/{MAX_RETRIES}")
                    continue

            # Extract only the <summary> part, discard <thinking>
            summary = content
            if "<summary>" in content and "</summary>" in content:
                summary = content.split("<summary>")[1].split("</summary>")[0].strip()

            usage = response.usage
            print(f"  ✓ Done. Tokens: in={usage.prompt_tokens}, out={usage.completion_tokens}")
            return (arxiv_id, summary)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(5)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                return None


def summarize_papers():
    conn = init_db()
    unsaved = get_unsummarized_papers(conn)
    if not unsaved:
        print("No papers to summarize")
        conn.close()
        return

    print(f"Papers to summarize: {len(unsaved)}")
    results = []

    with ThreadPoolExecutor(max_workers=SUMMARY_PARALLEL) as executor:
        futures = {executor.submit(summarize_single, paper): paper for paper in unsaved}
        for future in as_completed(futures):
            paper = futures[future]
            print(f"\nDone: {paper['Title'][:70]}...")
            try:
                result = future.result()
            except Exception as e:
                print(f"  Unexpected error: {e}")
                continue
            if result:
                results.append(result)
                update_summaries(conn, [result])
                print(f"  [saved to DB]")

    if results:
        update_summaries(conn, results)
        print(f"\nSaved {len(results)} summaries to DB")

    conn.close()

# ── Generate README ─────────────────────────────────────────────────────

import re

def clean_summary_tags(text: str) -> str:
    """Strip <summary>...</summary> wrapper if present, leave inner content."""
    # Full pair match
    m = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Unclosed <summary> at start (truncated output)
    if text.startswith('<summary>'):
        return text[len('<summary>'):].strip()
    # Stray closing tag
    if text.startswith('</summary>'):
        return text[len('</summary>'):].strip()
    return text

def escape_md(text: str) -> str:
    """Escape pipe characters for markdown tables."""
    return text.replace("|", "\\|").replace("\n", "<br>")

def generate_readme():
    conn = init_db()
    beijing_tz = pytz.timezone("Asia/Shanghai")
    current_date = datetime.now(beijing_tz).strftime("%Y-%m-%d")

    f_rm = open("README.md", "w")
    f_rm.write("# Daily Papers\n")
    f_rm.write("Daily arXiv papers filtered and ranked by LLM relevance.\n\n")
    f_rm.write(f"Last update: {current_date}\n\n")

    for section, config in KEYWORDS.items():
        max_papers = config.get("max_papers_per_section", CONFIG_DEFAULTS["max_papers_per_section"])
        papers = get_section_papers(conn, section, max_papers)
        print(f"  {section}: {len(papers)} papers kept")

        if not papers:
            f_rm.write(f"## {section}\n\n*No relevant papers found.*\n\n")
            continue

        # Build table: Date | AI Summary
        f_rm.write(f"## {section}\n\n")
        f_rm.write("| **Date** | **AI Summary** |\n")
        f_rm.write("| --- | --- |\n")

        for p in papers:
            date_str = p["Date"].split("T")[0] if "T" in str(p["Date"]) else str(p["Date"])
            summary_text = p.get("Summary") or "Not yet summarized"
            summary_text = clean_summary_tags(summary_text)
            summary = escape_md(summary_text)
            title_link = f"[{p['Title']}]({p['Link']})"
            if summary_text and summary_text != "Not yet summarized":
                cell = f"**{title_link}**<br><details><summary>Show Summary</summary>{summary}</details>"
            else:
                cell = f"**{title_link}**<br>{summary}"
            f_rm.write(f"| {date_str} | {cell} |\n")

        f_rm.write("\n")

    f_rm.close()
    conn.close()
    print("README regenerated")

# ── Main ────────────────────────────────────────────────────────────────

def usage():
    print("Usage: python main.py {fetch|screen|summarize|readme|all}")
    print("  fetch      - fetch new papers from arXiv and save to DB")
    print("  screen     - screen all unscreened papers (batch=5, parallel=8)")
    print("  summarize  - generate AI summaries for kept papers (v4-pro)")
    print("  readme     - regenerate README from DB")
    print("  all        - fetch + screen + summarize + readme")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()

    cmd = sys.argv[1]

    if cmd == "fetch":
        fetch_papers()
    elif cmd == "screen":
        screen_papers()
    elif cmd == "summarize":
        summarize_papers()
    elif cmd == "readme":
        generate_readme()
    elif cmd == "all":
        fetch_papers()
        screen_papers()
        summarize_papers()
        generate_readme()
    else:
        usage()
