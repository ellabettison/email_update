#!/usr/bin/env python3
"""Weekly AI-safety reading digest.

Free to run: GitHub Actions does the scheduling, Gmail SMTP sends the email.
Usage:  python digest.py            (sends email)
        python digest.py --dry-run  (writes digest.html, sends nothing)

How items are chosen
  relevance : keyword score. Model-psychology terms weigh most, then core safety
              terms. Generic ML terms (RLHF, fine-tuning, "alignment" on its own)
              never make an item eligible by themselves.
  quality   : source bonus, trusted-author bonus, Hugging Face upvotes (only
              counted when the item is actually on-topic).
  recency   : newer items get a ranking bonus; older ones can still win if strong.
  layout    : "Read these first" (best few), "Also worth a look", and a small
              "Training & methods" slot for popular training-technique papers.

Window: every run considers the last 30 days. Only items actually shown are
marked as seen, so strong near-misses carry over to later weeks and quiet weeks
are topped up from the backlog. Because of that the run time/cadence does not
matter (any gap up to 30 days is covered).
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
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import feedparser

# ---------------------------------------------------------------- config ---
TOP_PICKS = 5             # "Read these first"
MAX_MORE = 8              # "Also worth a look"
MAX_TRAINING = 2          # "Training & methods"
MAX_PER_SOURCE_TOP = 3    # so one blog can't fill the top section
MAX_PER_SOURCE_MORE = 4
MAX_AGE_HOURS = 30 * 24   # candidate window
RECENT_DAYS, RECENT_BONUS = 7, 3    # ranking bonus: published in the last week
MID_DAYS, MID_BONUS = 14, 1         # ... or the last two weeks
MIN_RELEVANCE = 3         # needed (together with a core term) to be eligible
AUTHOR_BONUS = 8          # per trusted author on a paper (max 2 counted)
UPVOTE_STEP = 20          # +1 ranking point per this many HF upvotes ...
UPVOTE_BONUS_MAX = 4      # ... up to this many, and only if the paper is on-topic
TRAINING_MIN_UPVOTES = 60 # popular training-technique papers get the small slot
TRAINING_MIN_SCORE = 3
HF_LOOKBACK_DAYS = 30
PAGE_MAX_NEW = 10         # newest N links on a no-feed index page are considered
ARXIV_PAGE = 100          # results per arXiv API request
ARXIV_MAX_PAGES = 4       # pages per keyword query (400 newest matches)
SEEN_FILE = Path("seen.json")
UA = "Mozilla/5.0 (compatible; safety-digest/1.0; personal feed reader)"

# (name, url, always_include, source_bonus)
# always_include=True  -> every new item is eligible (low volume, high signal)
# always_include=False -> the item must also match your keywords
# Bonus only affects ranking, never eligibility.
# Verified live: LessWrong feed format. view=curated and karmaThreshold are
# documented as supported (thresholds snap to 2/30/45/75/125/200).
# Unverified: the Alignment Forum URL. Broken feeds are listed in the email.
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

# Sites with no RSS feed: read the index page, take the newest PAGE_MAX_NEW
# links matching link_regex, read each article's title/description/date.
# (name, index_url, link_regex, always_include, source_bonus)
PAGES = [
    ("Anthropic Alignment Science", "https://alignment.anthropic.com/",
     r"^https://alignment\.anthropic\.com/\d{4}/", True, 9),
]

HF_BASE = "https://huggingface.co/api/daily_papers"

# Model psychology / behaviour. Title hit = 6, abstract/body hit = 3.
PRIORITY_KEYWORDS = [
    "sycophancy", "sycophantic", "persona", "deception", "deceptive", "alignment faking",
    "jailbreak", "jailbreaking", "sandbagging", "introspection", "emergent misalignment",
    "agentic misalignment", "reward hacking", "scheming", "self-preservation",
    "situational awareness", "evaluation awareness", "model psychology", "model character",
    "character training", "honesty", "persuasion", "role-play", "roleplay", "unfaithful",
    "backdoor", "sleeper agent", "model organism", "lie detector", "lie detection",
    "manipulative", "user manipulation",
]
# Core safety terms. Title hit = 4, abstract/body hit = 2.
SAFETY_KEYWORDS = [
    "ai safety", "misalignment", "interpretability", "mechanistic interpretability",
    "sparse autoencoder", "activation steering", "red teaming", "red-teaming",
    "ai control", "oversight", "scalable oversight", "dangerous capabilities",
    "monitorability", "cot monitoring", "chain-of-thought monitoring", "ai monitor",
    "cot faithfulness", "chain-of-thought faithfulness", "safety training",
    "constitutional ai", "model spec", "alignment research", "alignment training",
]
# Training-technique terms. Title hit = 2, body hit = 1. Never sufficient alone
# for ordinary eligibility; they only qualify popular papers for the small slot.
TRAINING_KEYWORDS = [
    "alignment", "rlhf", "rlvr", "rlaif", "reinforcement learning", "post-training",
    "preference optimization", "dpo", "reward model", "fine-tuning", "distillation",
    "on-policy", "chain of thought", "chain-of-thought",
]

ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]

# Researchers whose safety work is usually worth reading. My judgement, biased
# toward model behaviour/psychology, then interpretability, then control/evals.
# Matching is on exact full names as arXiv prints them. Edit freely.
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


def clean_title(title, url=""):
    """Drop unrendered template placeholders such as '{title}'."""
    t = re.sub(r"\{[^{}]*\}", "", title or "")
    t = re.sub(r"\s+", " ", t).strip()
    if not t and url:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        t = slug.replace("-", " ").replace("_", " ").title()
    return t


def kw_regex(kw):
    return re.compile(r"\b" + re.escape(kw) + r"s?\b", re.I)


PRIORITY_RE = [kw_regex(k) for k in PRIORITY_KEYWORDS]
SAFETY_RE = [kw_regex(k) for k in SAFETY_KEYWORDS]
TRAINING_RE = [kw_regex(k) for k in TRAINING_KEYWORDS]


def norm_name(n):
    n = n.lower().replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


TRUSTED_NORM = {norm_name(a) for a in TRUSTED_AUTHORS}


def trusted_in(entry):
    names = [a.get("name", "") for a in entry.get("authors", [])]
    return [n for n in names if norm_name(n) in TRUSTED_NORM]


def score(title, text):
    """-> (relevance, has_core_term, training_score)."""
    rel, strong, train = 0, False, 0
    for rx in PRIORITY_RE:
        if rx.search(title):
            rel, strong = rel + 6, True
        elif rx.search(text):
            rel, strong = rel + 3, True
    for rx in SAFETY_RE:
        if rx.search(title):
            rel, strong = rel + 4, True
        elif rx.search(text):
            rel, strong = rel + 2, True
    for rx in TRAINING_RE:
        if rx.search(title):
            train += 2
        elif rx.search(text):
            train += 1
    return rel, strong, train


def judge(rel, strong, train, always=False, trusted=False, upvotes=None):
    """-> (eligible, training_only)."""
    if always:
        return True, False
    if strong and rel >= MIN_RELEVANCE:
        return True, False
    if trusted and strong:
        return True, False
    if upvotes is not None and upvotes >= TRAINING_MIN_UPVOTES and train >= TRAINING_MIN_SCORE:
        return True, True
    return False, False


def entry_time(e):
    for key in ("published_parsed", "updated_parsed"):
        t = e.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


# ----------------------------------------------------------------- dates ---
_MON = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON_RE = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"


def fmt_date(dt):
    return f"{dt.day} {dt:%b %Y}" if dt else ""


def parse_iso(stamp):
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def display_to_datetime(text):
    """'4 Oct 2026' -> datetime; 'Oct 2026' -> late in that month; year-only -> None."""
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]{3}) (\d{4})", text or "")
    if m and m.group(2).lower() in _MON:
        return _mk_dt(m.group(3), _MON[m.group(2).lower()], m.group(1))
    m = re.fullmatch(r"([A-Za-z]{3}) (\d{4})", text or "")
    if m and m.group(1).lower() in _MON:
        return _mk_dt(m.group(2), _MON[m.group(1).lower()], 28)
    return None


def _mk_dt(y, m, d):
    try:
        return datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
    except ValueError:
        return None


def _mk(y, m, d):
    try:
        return fmt_date(datetime(int(y), int(m), int(d)))
    except ValueError:
        return ""


def parse_full_date(text):
    """Day-precision dates only: ISO, '4 Oct 2026', 'Oct 4, 2026'."""
    text = text or ""
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return _mk(m.group(1), m.group(2), m.group(3))
    m = re.search(rf"\b(\d{{1,2}})\s+{_MON_RE},?\s+(\d{{4}})\b", text, re.I)
    if m:
        return _mk(m.group(3), _MON[m.group(2).lower()], m.group(1))
    m = re.search(rf"\b{_MON_RE}\s+(\d{{1,2}}),?\s+(\d{{4}})\b", text, re.I)
    if m:
        return _mk(m.group(3), _MON[m.group(1).lower()], m.group(2))
    return ""


def parse_date_str(text):
    full = parse_full_date(text)
    if full:
        return full
    m = re.search(rf"\b{_MON_RE}\s+(\d{{4}})\b", text or "", re.I)
    if m:
        return f"{m.group(1).title()[:3]} {m.group(2)}"
    return ""


DATE_META_KEYS = [
    "article:published_time", "og:article:published_time", "citation_publication_date",
    "citation_date", "dc.date", "dcterms.date", "date", "pubdate", "publish_date",
    "published_time",
]


def page_date(ap, raw_html, index_nodes, url):
    """Best-effort publication date for a page that has no feed. Returns text."""
    for k in DATE_META_KEYS:
        d = parse_date_str(ap.meta.get(k))
        if d:
            return d
    for t in ap.times:
        d = parse_date_str(t)
        if d:
            return d
    m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw_html or "")
    if m and parse_date_str(m.group(1)):
        return parse_date_str(m.group(1))
    for n in index_nodes:                       # date shown beside the link on the index
        d = parse_date_str(n)
        if d:
            return d
    d = parse_full_date(ap.text_head)           # a full date near the top of the article
    if d:
        return d
    m = re.search(r"/(20\d{2})/", url)          # last resort: year from the URL
    return m.group(1) if m else ""


# ------------------------------------------------------- html / http utils ---
class _LinkMeta(HTMLParser):
    """Collects links (with their text), <meta>, <title>, <time>, paragraphs."""

    def __init__(self):
        super().__init__()
        self.links, self.meta, self.title, self._in_title = [], {}, "", False
        self.anchors, self._href = {}, None      # href -> text nodes inside the link
        self.paragraphs, self._p = [], None
        self.times = []                          # <time datetime="...">
        self.text_head, self._skip = "", 0       # first ~2500 chars of visible text

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
            self._href = a["href"]
            self.anchors.setdefault(self._href, [])
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, a["content"])
        elif tag == "title":
            self._in_title = True
        elif tag == "time" and a.get("datetime"):
            self.times.append(a["datetime"])
        elif tag == "p":
            self._p = []

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        elif tag == "a":
            self._href = None
        elif tag == "p" and self._p is not None:
            txt = " ".join("".join(self._p).split())
            if txt:
                self.paragraphs.append(txt)
            self._p = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        text = " ".join(data.split())
        if text:
            if self._href is not None:
                self.anchors[self._href].append(text)
            if len(self.text_head) < 2500:
                self.text_head += text + " "
        if self._p is not None:
            self._p.append(data)


def http_get(url):
    req = Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/rss+xml, application/atom+xml, application/xml, "
                  "application/json, text/html;q=0.9, */*;q=0.8",
    })
    with urlopen(req, timeout=30) as r:
        text = r.read().decode(r.headers.get_content_charset() or "utf-8", "replace")
        return text, r.geturl()


def norm_url(u):
    u = u.split("#")[0]
    u = re.sub(r"/index\.html$", "/", u)
    return u.rstrip("/")


def anchor_description(nodes, title):
    """Index pages often put title + blurb inside the link; take the blurb."""
    t = title.lower()
    cands = [n for n in nodes if n.lower() != t]
    if not cands:
        return ""
    best = max(cands, key=len)
    if best.lower().startswith(t):
        best = best[len(title):].lstrip(" -\u2013\u2014:|")
        # title + "Author, Author, 2026" + blurb squashed into one text node:
        # drop the author/year run (short, no full stop, ends ", YYYY").
        best = re.sub(r"^[^.]{0,150}?,\s*20\d{2}\s+(?=[A-Z])", "", best)
    return best


def first_paragraph(paragraphs, min_len=120):
    return next((p for p in paragraphs if len(p) >= min_len), "")


# ---------------------------------------------------------------- sources ---
def arxiv_url(search_query):
    return (
        "https://export.arxiv.org/api/query?search_query=" + quote(search_query)
        + f"&sortBy=submittedDate&sortOrder=descending&max_results={ARXIV_PAGE}"
    )


def arxiv_keyword_urls(chunk=19):
    """Several short queries rather than one huge one."""
    terms = PRIORITY_KEYWORDS + SAFETY_KEYWORDS
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    urls = []
    for i in range(0, len(terms), chunk):
        kw = " OR ".join(f'all:"{t}"' for t in terms[i:i + chunk])
        urls.append(arxiv_url(f"({cats}) AND ({kw})"))
    return urls


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


def parse_feed(url, retries=0):
    """feedparser with retries, a plain-fetch fallback, and a useful diagnosis."""
    feed, diag = None, ""
    for attempt in range(retries + 1):
        feed = feedparser.parse(url, agent=UA)
        if feed.entries:
            return feed, ""
        diag = f"HTTP {feed.get('status', 'n/a')}"
        exc = feed.get("bozo_exception")
        if exc:
            diag += f", {type(exc).__name__}: {str(exc)[:80]}"
        if attempt < retries:
            time.sleep(10 * (attempt + 1))
    if url.startswith("http"):       # some hosts reject feedparser's own fetch
        try:
            text, _ = http_get(url)
            alt = feedparser.parse(text)
            if alt.entries:
                return alt, ""
        except Exception as exc:
            diag += f"; plain fetch failed: {exc}"
    return feed, diag


def entries_to_items(entries, name, always, bonus, is_paper, seen, cutoff):
    items = []
    for e in entries:
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
        title = clean_title(clean(e.get("title"), 200), e.get("link", ""))
        # Some feeds (LessWrong) ship the whole post: score the opening ~1500
        # chars, but only show 320.
        rel, strong, train = score(title, clean(raw, 1500))
        trusted = trusted_in(e) if is_paper else []
        ok, training_only = judge(rel, strong, train, always, bool(trusted))
        if not ok:
            continue
        extra = AUTHOR_BONUS * min(len(trusted), 2) if is_paper else bonus
        items.append({
            "id": uid, "source": name, "title": title, "summary": clean(raw, 320),
            "link": e.get("link", uid), "score": rel + min(train, 3) + extra,
            "strong": strong, "training_only": training_only, "trusted": trusted,
            "upvotes": None, "date": fmt_date(when), "when": when,
        })
    return items


def fetch(name, url, always, bonus, is_paper, seen, cutoff, errors, retries=0):
    try:
        feed, diag = parse_feed(url, retries)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        return []
    if not feed.entries:
        errors.append(f"{name}: no entries ({diag}) - {url[:110]}")
        return []
    return entries_to_items(feed.entries, name, always, bonus, is_paper, seen, cutoff)


def fetch_arxiv(name, url, seen, cutoff, errors, max_pages):
    """Page back through an arXiv API query until it is older than the cutoff.
    Returns (items, first_page_ok)."""
    items = []
    for page in range(max_pages):
        if page:
            time.sleep(5)               # arXiv asks for spacing between calls
        feed, diag = parse_feed(f"{url}&start={page * ARXIV_PAGE}", retries=2)
        if not feed.entries:
            if page == 0:
                errors.append(f"{name}: no entries ({diag}) - {url[:110]}")
                return items, False
            break
        items += entries_to_items(feed.entries, name, False, 0, True, seen, cutoff)
        times = [t for t in (entry_time(e) for e in feed.entries) if t]
        if len(feed.entries) < ARXIV_PAGE or (times and min(times) < cutoff):
            break
    return items, True


def fetch_page_source(name, index_url, link_re, always, bonus, seen, errors, cutoff=None):
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
        found.append((nid, full, href))
    if not found:
        errors.append(f"{name}: no article links matched on {index_url} (site layout changed?)")
        return []
    items = []
    # Only the newest PAGE_MAX_NEW links are ever considered, so older posts
    # can never trickle in one batch at a time.
    for nid, full, href in [t for t in found[:PAGE_MAX_NEW] if t[0] not in seen]:
        try:
            art, _ = http_get(full)
        except Exception as exc:
            errors.append(f"{name}: could not load {full} ({exc})")
            continue
        time.sleep(1)
        ap = _LinkMeta()
        ap.feed(art)
        title = clean_title(clean(ap.meta.get("og:title") or ap.title, 200), full)
        nodes = p.anchors.get(href, [])
        date = page_date(ap, art, nodes, full)
        when = display_to_datetime(date)
        if cutoff and when and when < cutoff:
            continue
        # description: page meta -> blurb on the index page -> first real paragraph
        desc = ap.meta.get("og:description") or ap.meta.get("description") or ""
        if len(clean(desc)) < 40:
            desc = anchor_description(nodes, title)
        if len(clean(desc)) < 40:
            desc = first_paragraph(ap.paragraphs)
        rel, strong, train = score(title, clean(desc, 1500))
        ok, training_only = judge(rel, strong, train, always)
        if not ok:
            continue
        items.append({
            "id": nid, "source": name, "title": title, "summary": clean(desc, 320),
            "link": full, "score": rel + min(train, 3) + bonus, "strong": strong,
            "training_only": training_only, "trusted": [], "upvotes": None,
            "date": date, "when": when,
        })
    return items


def enrich_descriptions(items):
    """Fill in missing/very short descriptions (and dates) from the article page."""
    for it in items:
        need_desc = len(it["summary"]) < 100
        need_date = not it.get("date")
        if not (need_desc or need_date) or not it["link"].startswith("http"):
            continue
        try:
            art, _ = http_get(it["link"])
        except Exception:
            continue
        time.sleep(1)
        ap = _LinkMeta()
        ap.feed(art)
        if need_desc:
            cand = clean(ap.meta.get("og:description") or ap.meta.get("description")
                         or first_paragraph(ap.paragraphs), 320)
            if len(cand) > len(it["summary"]):
                it["summary"] = cand
        if need_date:
            it["date"] = page_date(ap, art, [], it["link"])


def fetch_hf_papers(seen, now, errors):
    """Hugging Face daily papers: community upvotes as a quality signal.
    Daily lists skip weekends, so ask for the latest list plus each of the last
    HF_LOOKBACK_DAYS dates."""
    urls = [f"{HF_BASE}?limit=100"] + [
        f"{HF_BASE}?date={(now - timedelta(days=d)):%Y-%m-%d}&limit=100"
        for d in range(1, HF_LOOKBACK_DAYS + 1)
    ]
    rows, ok_calls, fails, last_err = [], 0, 0, ""
    for i, url in enumerate(urls):
        if i:
            time.sleep(1)
        try:
            text, _ = http_get(url)
            data = json.loads(text)
        except Exception as exc:
            last_err = str(exc)
            fails += 1
            if not ok_calls and fails >= 3:   # HF unreachable: stop early
                break
            continue
        if isinstance(data, list):
            ok_calls += 1
            rows += data
    if not ok_calls:
        errors.append(f"Hugging Face daily papers: all requests failed ({last_err})")
        return []
    cutoff = now - timedelta(hours=MAX_AGE_HOURS)
    items, done = [], set()
    for row in rows:
        p = row.get("paper") or {}
        pid = p.get("id")
        if not pid or pid in done or f"arxiv:{pid}" in seen:
            continue
        done.add(pid)
        when = parse_iso(p.get("submittedOnDailyAt") or row.get("publishedAt"))
        if when and when < cutoff:
            continue
        title = clean_title(clean(p.get("title") or row.get("title"), 200))
        abstract = clean(p.get("summary") or row.get("summary"), 1500)
        upv = p.get("upvotes") or 0
        trusted = [a["name"] for a in p.get("authors", [])
                   if norm_name(a.get("name", "")) in TRUSTED_NORM]
        rel, strong, train = score(title, abstract)
        ok, training_only = judge(rel, strong, train, False, bool(trusted), upv)
        if not ok:
            continue
        bonus = AUTHOR_BONUS * min(len(trusted), 2)
        if strong:
            bonus += min(upv // UPVOTE_STEP, UPVOTE_BONUS_MAX)
        items.append({
            "id": f"arxiv:{pid}", "source": "Hugging Face daily papers",
            "title": title, "summary": clean(abstract, 320),
            "link": f"https://arxiv.org/abs/{pid}", "score": rel + min(train, 3) + bonus,
            "strong": strong, "training_only": training_only, "trusted": trusted,
            "upvotes": upv, "when": when,
            "date": fmt_date(parse_iso(row.get("publishedAt")) or when),
        })
    return items


# -------------------------------------------------------------- selection ---
def merge(items):
    """Same post can arrive via several feeds/queries; keep the best-scored."""
    best = {}
    for it in items:
        cur = best.get(it["id"])
        if cur is None or it["score"] > cur["score"]:
            best[it["id"]] = it
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def take(items, n, per_source):
    out, counts = [], {}
    for it in items:
        if len(out) >= n:
            break
        if counts.get(it["source"], 0) >= per_source:
            continue
        counts[it["source"]] = counts.get(it["source"], 0) + 1
        out.append(it)
    return out


def recency_bonus(when, now):
    if not when:
        return 0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = now - when
    if age <= timedelta(days=RECENT_DAYS):
        return RECENT_BONUS
    if age <= timedelta(days=MID_DAYS):
        return MID_BONUS
    return 0


def select(items, now=None):
    now = now or datetime.now(timezone.utc)
    pool = merge(items)
    for it in pool:
        it["rank"] = it["score"] + recency_bonus(it.get("when"), now)
    pool.sort(key=lambda x: x["rank"], reverse=True)
    top = take([i for i in pool if i["strong"] and not i["training_only"]],
               TOP_PICKS, MAX_PER_SOURCE_TOP)
    used = {i["id"] for i in top}
    rest = [i for i in pool if i["id"] not in used]
    more = take([i for i in rest if not i["training_only"]], MAX_MORE, MAX_PER_SOURCE_MORE)
    training = take([i for i in rest if i["training_only"]], MAX_TRAINING, MAX_TRAINING)
    return top, more, training, pool


# ----------------------------------------------------------------- output ---
def render(top, more, training, errors):
    today = datetime.now(timezone.utc).strftime("%a %d %b %Y")

    def item_html(it, n):
        badge = ""
        if it["trusted"]:
            badge = (" &middot; <span style='color:#b8860b'>&#9733; "
                     + html.escape(", ".join(it["trusted"][:3])) + "</span>")
        if it.get("upvotes"):
            badge += f" &middot; <span style='color:#555'>&#9650; {it['upvotes']}</span>"
        date_html = (f" &middot; {html.escape(it['date'])}" if it.get("date")
                     else " &middot; <span style='color:#999'>date unknown</span>")
        return (
            "<div style='margin:0 0 14px'>"
            f"<a href='{html.escape(it['link'])}' style='font-weight:600;"
            f"text-decoration:none'>{html.escape(it['title'])}</a>"
            f"<div style='color:#666;font-size:12px'>{html.escape(it['source'])}{date_html}{badge}</div>"
            f"<div style='font-size:14px;margin-top:2px'>{html.escape(clean(it['summary'], n))}</div>"
            "</div>"
        )

    def block(heading, items, n):
        if not items:
            return ""
        return (f"<h3 style='margin:26px 0 8px'>{heading}</h3>"
                + "".join(item_html(i, n) for i in items))

    body = (block("Read these first", top, 320)
            + block("Also worth a look", more, 200)
            + block("Training &amp; methods", training, 180))
    if not body:
        body = "<p>Nothing new matched this week.</p>"
    err = ""
    if errors:
        err = ("<hr><p style='color:#a00;font-size:12px'>Feed problems:<br>"
               + "<br>".join(html.escape(x) for x in errors) + "</p>")
    page = (
        "<div style='font-family:-apple-system,Segoe UI,sans-serif;max-width:640px'>"
        f"<h2 style='margin:0'>Safety digest &middot; weekly</h2><div style='color:#666'>{today}</div>"
        f"{body}{err}</div>"
    )
    text_lines = [f"Safety digest (weekly) - {today}", ""]
    for heading, items in (("READ THESE FIRST", top), ("ALSO WORTH A LOOK", more),
                           ("TRAINING & METHODS", training)):
        if items:
            text_lines += [heading, ""]
            for it in items:
                text_lines += [it["title"],
                               f"  {it['source']} - {it.get('date') or 'date unknown'}",
                               f"  {it['link']}", f"  {it['summary']}", ""]
    return page, "\n".join(text_lines)


def send(page, text, n_items):
    addr = os.environ["EMAIL_ADDRESS"]
    pwd = os.environ["EMAIL_APP_PASSWORD"]
    msg = EmailMessage()
    msg["Subject"] = f"Safety digest (weekly): {n_items} items"
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
    errors, items = [], []

    for name, url, always, bonus in FEEDS:
        items += fetch(name, url, always, bonus, False, seen, cutoff, errors)
    for name, url, link_re, always, bonus in PAGES:
        items += fetch_page_source(name, url, link_re, always, bonus, seen, errors, cutoff)

    jobs = [("arXiv", u, ARXIV_MAX_PAGES) for u in arxiv_keyword_urls()] + \
           [("arXiv (trusted authors)", u, 3) for u in arxiv_author_urls()]
    failed_in_a_row = 0
    for i, (name, url, pages) in enumerate(jobs):
        if i:
            time.sleep(5)
        got, ok = fetch_arxiv(name, url, seen, cutoff, errors, pages)
        items += got
        failed_in_a_row = 0 if ok else failed_in_a_row + 1
        if failed_in_a_row >= 2:
            errors.append("arXiv: two queries in a row failed, skipping the rest this run")
            break
    items += fetch_hf_papers(seen, now, errors)

    top, more, training, pool = select(items, now)
    chosen = top + more + training
    enrich_descriptions(chosen)
    page, text = render(top, more, training, errors)
    if dry:
        Path("digest.html").write_text(page)
        print(f"dry run: {len(top)} top, {len(more)} more, {len(training)} training, "
              f"{len(pool)} eligible in window, {len(errors)} feed errors")
        return

    send(page, text, len(chosen))

    # Only what was actually shown counts as seen; strong near-misses stay
    # eligible (until they age out of the 30-day window) and carry over.
    for it in chosen:
        seen[it["id"]] = now.isoformat()
    keep_after = (now - timedelta(days=180)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= keep_after}
    SEEN_FILE.write_text(json.dumps(seen, indent=1))


if __name__ == "__main__":
    main()

