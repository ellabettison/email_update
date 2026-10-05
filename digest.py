#!/usr/bin/env python3
"""Daily AI-safety reading digest.

Free to run: GitHub Actions does the scheduling, Gmail SMTP sends the email.
Usage:  python digest.py            (sends email)
        python digest.py --dry-run  (writes digest.html, sends nothing)

Ranking = keyword relevance + quality proxies:
  * per-source bonus (trusted blogs/curators outrank generic feeds)
  * trusted-author bonus for arXiv papers
  * karma thresholds on LessWrong / Alignment Forum feeds
Keyword relevance is always required, so a trusted source never pushes in
something off-topic.
"""
import html
import json
import os
import re
import smtplib
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import feedparser

# ---------------------------------------------------------------- config ---
MAX_BLOG_ITEMS = 8
MAX_PAPERS = 8
MAX_AGE_HOURS = 72
MIN_SCORE_BLOG = 2        # keyword score needed from non-curated feeds
MIN_SCORE_PAPER = 3       # keyword score needed for arXiv papers by unknown authors
TRUSTED_MIN_KW = 2        # keyword score needed for papers by trusted authors
AUTHOR_BONUS = 8          # per trusted author on a paper (max 2 counted)
SEEN_FILE = Path("seen.json")

# (name, url, always_include, source_bonus)
# always_include=True  -> every new item is eligible (low volume, high signal)
# always_include=False -> item must also match your keywords (MIN_SCORE_BLOG)
# Bonus only affects ranking, never eligibility.
# Verified live: the LessWrong feed format (view=frontpage). view=curated and
# karmaThreshold are documented as officially supported; karmaThreshold snaps
# to 2/30/45/75/125/200. Unverified: the Alignment Forum URL (same codebase).
# Broken feeds are listed at the bottom of the email.
FEEDS = [
    ("LessWrong (curated)", "https://www.lesswrong.com/feed.xml?view=curated", True, 6),
    ("LessWrong (karma 45+)", "https://www.lesswrong.com/feed.xml?view=frontpage&karmaThreshold=45", False, 3),
    ("Alignment Forum (karma 30+)", "https://www.alignmentforum.org/feed.xml?karmaThreshold=30", False, 4),
    ("Transformer Circuits", "https://transformer-circuits.pub/feed.xml", True, 7),
    ("Anthropic Research", "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml", False, 5),
    ("Anthropic Frontier Red Team", "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_red.xml", False, 3),
    ("Google DeepMind", "https://deepmind.google/blog/rss.xml", False, 2),
    ("OpenAI Research", "https://openai.com/blog/rss.xml", False, 2),
    ("Redwood Research", "https://blog.redwoodresearch.org/feed", True, 8),
    ("METR", "https://metr.org/feed.xml", True, 8),
    ("AI Safety Newsletter", "https://newsletter.safe.ai/feed", True, 5),
    ("Import AI", "https://importai.substack.com/feed", True, 4),
    ("Transformer", "https://www.transformernews.ai/feed", True, 4),
    ("Interconnects", "https://www.interconnects.ai/feed", True, 4),
    ("Zvi", "https://thezvi.substack.com/feed", False, 3),
    ("Raschka", "https://magazine.sebastianraschka.com/feed", False, 2),
    ("Hugging Face blog", "https://huggingface.co/blog/feed.xml", False, 0),
]

# Sites with no RSS feed: the script reads the index page, finds links that match
# link_regex, and reads each new article's title/description from its own page.
# (name, index_url, link_regex, always_include, source_bonus)
PAGES = [
    ("Anthropic Alignment Science", "https://alignment.anthropic.com/",
     r"^https://alignment\.anthropic\.com/\d{4}/", True, 9),
]
HF_URL = "https://huggingface.co/api/daily_papers?limit=100"
HF_MAX_AGE_HOURS = 120    # daily papers skip weekends, so look back further
HF_STRONG_UPVOTES = 25    # this many upvotes lowers the keyword bar by 1
PAGE_MAX_NEW = 6          # articles inspected per page per run (newest first)

# Model psychology / behaviour: weighted higher.
PRIORITY_KEYWORDS = [
    "sycophancy", "persona", "deception", "alignment faking", "jailbreak",
    "sandbagging", "introspection", "emergent misalignment", "reward hacking",
    "scheming", "manipulative", "user manipulation", "situational awareness", "self-knowledge",
    "model psychology", "model character", "character training", "honesty", "persuasion", "role-play",
    "chain of thought", "chain-of-thought", "unfaithful", "backdoor",
]
# Wider safety / training coverage.
BROAD_KEYWORDS = [
    "alignment", "ai safety", "interpretability", "mechanistic", "sparse autoencoder",
    "steering", "probing", "red teaming", "red-teaming", "ai control", "oversight",
    "rlhf", "rlvr", "rlaif", "reinforcement learning", "post-training",
    "preference optimization", "dpo", "constitutional", "evaluation",
    "dangerous capabilities", "misalignment", "robustness", "adversarial",
    "monitoring", "scalable oversight", "reward model", "fine-tuning",
]

ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]

# Researchers whose safety work is usually worth reading. My judgement, biased
# toward model behaviour/psychology, then interpretability, then control/evals.
# Matching is on exact full names as arXiv prints them, so initials-only or
# variant spellings will be missed. Edit freely.
TRUSTED_AUTHORS = [
    # behaviour, persona, deception
    "Owain Evans", "Jan Betley", "James Chua", "Evan Hubinger", "Ethan Perez",
    "Samuel R. Bowman", "Jack Lindsey", "Sam Marks", "Jan Leike", "Miles Turpin",
    "Cem Anil", "Carson Denison", "Monte MacDiarmid", "Mrinank Sharma",
    "Marius Hobbhahn", "Mikita Balesni", "Alexander Meinke", "Murray Shanahan",
    "Nina Panickssery", "Alexander Matt Turner", "Micah Carroll",
    "Maksym Andriushchenko", "Javier Rando", "Stephen Casper", "Boaz Barak",
    # interpretability
    "Neel Nanda", "Senthooran Rajamanoharan", "Arthur Conmy", "Chris Olah",
    "Joshua Batson", "Adly Templeton", "Trenton Bricken", "Lee Sharkey",
    "David Bau", "Atticus Geiger", "Jacob Steinhardt", "Sarah Schwettmann",
    "Yonatan Belinkov", "Mor Geva", "Jacob Andreas", "Been Kim", "Andy Zou",
    "Dan Hendrycks", "Mantas Mazeika", "Roger Grosse",
    # control, oversight, evals, theory
    "Buck Shlegeris", "Ryan Greenblatt", "Fabien Roger", "Beth Barnes",
    "Geoffrey Irving", "Paul Christiano", "Jacob Hilton", "Rohin Shah",
    "Mary Phuong", "Anca Dragan", "Victoria Krakovna", "Tom Everitt",
    "Zachary Kenton", "David Duvenaud", "Nicholas Carlini", "Julian Michael",
    "Akbir Khan", "Sebastian Farquhar", "Dylan Hadfield-Menell",
    # post-training
    "Nathan Lambert",
]
# ---------------------------------------------------------------------------


def clean(text, limit=300):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def kw_regex(kw):
    return re.compile(r"\b" + re.escape(kw) + r"\b", re.I)


PRIORITY_RE = [kw_regex(k) for k in PRIORITY_KEYWORDS]
BROAD_RE = [kw_regex(k) for k in BROAD_KEYWORDS]


def norm_name(n):
    n = n.lower().replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


TRUSTED_NORM = {norm_name(a) for a in TRUSTED_AUTHORS}


def trusted_in(entry):
    names = [a.get("name", "") for a in entry.get("authors", [])]
    return [n for n in names if norm_name(n) in TRUSTED_NORM]


def score(title, summary):
    s = 0
    for rx in PRIORITY_RE:
        if rx.search(title):
            s += 6
        elif rx.search(summary):
            s += 3
    for rx in BROAD_RE:
        if rx.search(title):
            s += 2
        elif rx.search(summary):
            s += 1
    return s


def entry_time(e):
    for key in ("published_parsed", "updated_parsed"):
        t = e.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


def arxiv_url(search_query):
    return (
        "https://export.arxiv.org/api/query?search_query=" + quote(search_query)
        + "&sortBy=submittedDate&sortOrder=descending&max_results=150"
    )


def arxiv_keyword_urls():
    terms = PRIORITY_KEYWORDS + [
        "alignment", "AI safety", "interpretability", "red teaming",
        "reward model", "RLHF", "post-training", "AI control", "misalignment",
    ]
    kw = " OR ".join(f'all:"{t}"' for t in terms)
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    return [arxiv_url(f"({cats}) AND ({kw})")]


def arxiv_author_urls(chunk=20):
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    urls = []
    for i in range(0, len(TRUSTED_AUTHORS), chunk):
        au = " OR ".join(f'au:"{a}"' for a in TRUSTED_AUTHORS[i:i + chunk])
        urls.append(arxiv_url(f"({cats}) AND ({au})"))
    return urls


def load_seen():
    if SEEN_FILE.exists():
        try:
            return json.loads(SEEN_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def fetch(name, url, always, bonus, is_paper, seen, cutoff, errors):
    items = []
    try:
        feed = feedparser.parse(url, agent="safety-digest/1.0")
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        return items
    if not feed.entries:
        errors.append(f"{name}: no entries returned (check URL) - {url[:120]}")
        return items
    for e in feed.entries:
        uid = e.get("id") or e.get("link")
        if is_paper and uid:
            m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", uid)
            if m:
                uid = "arxiv:" + m.group(1)  # same id as the Hugging Face source
        if not uid or uid in seen:
            continue
        when = entry_time(e)
        if when and when < cutoff:
            continue
        raw = e.get("summary") or e.get("description")
        title = clean(e.get("title"), 200)
        summary = clean(raw, 320)
        # Some feeds (LessWrong) ship the whole post; score on the opening
        # ~1500 chars, not just the 320 shown in the email.
        kw = score(title, clean(raw, 1500))

        trusted = trusted_in(e) if is_paper else []
        if is_paper:
            floor = TRUSTED_MIN_KW if trusted else MIN_SCORE_PAPER
            bonus_total = AUTHOR_BONUS * min(len(trusted), 2)
        else:
            floor = 0 if always else MIN_SCORE_BLOG
            bonus_total = bonus
        if kw < floor:
            continue
        items.append({
            "id": uid, "source": name, "title": title, "summary": summary,
            "link": e.get("link", uid), "score": kw + bonus_total,
            "trusted": trusted, "when": when, "upvotes": None,
        })
    return items


def merge(items):
    """Same post can arrive via several feeds/queries; keep the best-scored."""
    best = {}
    for it in items:
        cur = best.get(it["id"])
        if cur is None or it["score"] > cur["score"]:
            best[it["id"]] = it
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


class _LinkMeta(HTMLParser):
    """Collects <a href>, <meta> tags and <title> from an HTML page."""

    def __init__(self):
        super().__init__()
        self.links, self.meta, self.title, self._in_title = [], {}, "", False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def http_get(url):
    req = Request(url, headers={"User-Agent": "safety-digest/1.0 (personal use)"})
    with urlopen(req, timeout=30) as r:
        text = r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")
        return text, r.geturl()


def norm_url(u):
    u = u.split("#")[0]
    u = re.sub(r"/index\.html$", "/", u)
    return u.rstrip("/")


def fetch_page_source(name, index_url, link_re, always, bonus, seen, errors):
    try:
        page, final_url = http_get(index_url)
    except Exception as exc:
        errors.append(f"{name}: could not load {index_url} ({exc})")
        return []
    p = _LinkMeta()
    p.feed(page)
    found, ids = [], set()
    for href in p.links:
        full = urljoin(final_url, href)
        nid = norm_url(full)
        if not re.search(link_re, full) or nid == norm_url(final_url) or nid in ids:
            continue
        ids.add(nid)
        found.append((nid, full))
    if not found:
        errors.append(f"{name}: no article links matched on {index_url} (site layout changed?)")
        return []
    items = []
    unseen = [(n, f) for n, f in found if n not in seen][:PAGE_MAX_NEW]
    for nid, full in unseen:
        try:
            art, _ = http_get(full)
        except Exception as exc:
            errors.append(f"{name}: could not load {full} ({exc})")
            continue
        time.sleep(1)
        ap = _LinkMeta()
        ap.feed(art)
        title = clean(ap.meta.get("og:title") or ap.title or full, 200)
        summary = clean(ap.meta.get("og:description") or ap.meta.get("description") or "", 320)
        kw = score(title, summary)
        if kw < (0 if always else MIN_SCORE_BLOG):
            continue
        items.append({"id": nid, "source": name, "title": title, "summary": summary,
                      "link": full, "score": kw + bonus, "trusted": [], "when": None,
                      "upvotes": None})
    return items


def fetch_hf_papers(seen, now, errors):
    """Hugging Face daily papers: community upvotes as a quality signal."""
    try:
        text, _ = http_get(HF_URL)
        data = json.loads(text)
    except Exception as exc:
        errors.append(f"Hugging Face daily papers: {exc}")
        return []
    if not isinstance(data, list) or not data:
        errors.append("Hugging Face daily papers: empty or unexpected response")
        return []
    cutoff = now - timedelta(hours=HF_MAX_AGE_HOURS)
    items = []
    for row in data:
        p = row.get("paper") or {}
        pid = p.get("id")
        if not pid or f"arxiv:{pid}" in seen:
            continue
        stamp = p.get("submittedOnDailyAt") or row.get("publishedAt")
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            when = None
        if when and when < cutoff:
            continue
        title = clean(p.get("title") or row.get("title"), 200)
        abstract = clean(p.get("summary") or row.get("summary"), 1500)
        upv = p.get("upvotes") or 0
        trusted = [a["name"] for a in p.get("authors", [])
                   if norm_name(a.get("name", "")) in TRUSTED_NORM]
        kw = score(title, abstract)
        floor = MIN_SCORE_PAPER - (1 if upv >= HF_STRONG_UPVOTES else 0)
        if trusted:
            floor = min(floor, TRUSTED_MIN_KW)
        if kw < floor:
            continue
        bonus = min(upv // 10, 6) + AUTHOR_BONUS * min(len(trusted), 2)
        items.append({
            "id": f"arxiv:{pid}", "source": "Hugging Face daily papers",
            "title": title, "summary": clean(abstract, 320),
            "link": f"https://arxiv.org/abs/{pid}", "score": kw + bonus,
            "trusted": trusted, "when": when, "upvotes": upv,
        })
    return items


def render(blogs, papers, errors):
    today = datetime.now(timezone.utc).strftime("%a %d %b %Y")

    def block(heading, items):
        if not items:
            return ""
        out = [f"<h3 style='margin:24px 0 8px'>{heading}</h3>"]
        for it in items:
            badge = ""
            if it["trusted"]:
                badge = (" &middot; <span style='color:#b8860b'>&#9733; "
                         + html.escape(", ".join(it["trusted"][:3])) + "</span>")
            if it.get("upvotes"):
                badge += f" &middot; <span style='color:#555'>&#9650; {it['upvotes']}</span>"
            out.append(
                "<div style='margin:0 0 14px'>"
                f"<a href='{html.escape(it['link'])}' style='font-weight:600;"
                f"text-decoration:none'>{html.escape(it['title'])}</a>"
                f"<div style='color:#666;font-size:12px'>{html.escape(it['source'])}{badge}</div>"
                f"<div style='font-size:14px;margin-top:2px'>{html.escape(it['summary'])}</div>"
                "</div>"
            )
        return "".join(out)

    body = block("Blogs & newsletters", blogs) + block("New papers (arXiv)", papers)
    if not body:
        body = "<p>Nothing new matched today.</p>"
    err = ""
    if errors:
        err = ("<hr><p style='color:#a00;font-size:12px'>Feed problems:<br>"
               + "<br>".join(html.escape(x) for x in errors) + "</p>")
    page = (
        "<div style='font-family:-apple-system,Segoe UI,sans-serif;max-width:640px'>"
        f"<h2 style='margin:0'>Safety digest</h2><div style='color:#666'>{today}</div>"
        f"{body}{err}</div>"
    )
    text_lines = [f"Safety digest - {today}", ""]
    for it in blogs + papers:
        text_lines += [it["title"], f"  {it['source']}", f"  {it['link']}", ""]
    return page, "\n".join(text_lines)


def send(page, text, n_items):
    addr = os.environ["EMAIL_ADDRESS"]
    pwd = os.environ["EMAIL_APP_PASSWORD"]
    msg = EmailMessage()
    msg["Subject"] = f"Safety digest: {n_items} items"
    msg["From"] = addr
    msg["To"] = addr
    msg.set_content(text)
    msg.add_alternative(page, subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(addr, pwd)
        s.send_message(msg)


def main():
    dry = "--dry-run" in sys.argv
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=MAX_AGE_HOURS)
    seen = load_seen()
    errors = []

    blogs = []
    for name, url, always, bonus in FEEDS:
        blogs += fetch(name, url, always, bonus, False, seen, cutoff, errors)

    for name, url, link_re, always, bonus in PAGES:
        blogs += fetch_page_source(name, url, link_re, always, bonus, seen, errors)

    papers = []
    arxiv_urls = [("arXiv", u) for u in arxiv_keyword_urls()] + \
                 [("arXiv (trusted authors)", u) for u in arxiv_author_urls()]
    for i, (name, url) in enumerate(arxiv_urls):
        if i:
            time.sleep(3)  # be polite to the arXiv API
        papers += fetch(name, url, False, 0, True, seen, cutoff, errors)

    papers += fetch_hf_papers(seen, now, errors)

    blogs = merge(blogs)[:MAX_BLOG_ITEMS]
    papers = merge(papers)[:MAX_PAPERS]

    page, text = render(blogs, papers, errors)
    if dry:
        Path("digest.html").write_text(page)
        print(f"dry run: {len(blogs)} blog items, {len(papers)} papers, {len(errors)} feed errors")
        return

    send(page, text, len(blogs) + len(papers))

    for it in blogs + papers:
        seen[it["id"]] = now.isoformat()
    keep_after = (now - timedelta(days=30)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= keep_after}
    SEEN_FILE.write_text(json.dumps(seen, indent=1))


if __name__ == "__main__":
    main()
