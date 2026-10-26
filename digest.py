#!/usr/bin/env python3
"""Weekly AI-safety reading digest.

Free to run: GitHub Actions does the scheduling, Gmail SMTP sends the email.
Usage:  python digest.py            (sends email)
        python digest.py --dry-run  (writes digest.html, sends nothing)

How items are chosen
  relevance : keyword score. Model-psychology terms weigh most, then core safety
              terms. Generic ML terms (RLHF, fine-tuning, "alignment" on its own)
              never make an item eligible by themselves.
  quality   : source bonus; for papers an "outside approval" check (trusted
              author, Hugging Face upvotes, or a link from a trusted blog this
              month). Keyword relevance is capped so approval outweighs buzzwords.
  recency   : newer items get a ranking bonus; older ones can still win if strong.
  layout    : two balanced sections ("Blogs & newsletters" and "Papers"), each
              ranked best-first. Papers need outside approval for a main slot; a
              couple of very strong keyword matches without approval are shown
              separately as "Unvetted". Plus a small "Training & methods" slot and
              a "General AI" section for important non-safety papers/posts.

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
from urllib.parse import quote, urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import feedparser

# ---------------------------------------------------------------- config ---
POST_RECENT = ((7, 6), (14, 4), (30, 2))   # (max age in days, ranking bonus) for posts
POSTS_OLD_DAYS = 30       # a post older than this counts as "old" ...
POSTS_OLD_MAX = 1   # ... and at most this many old posts per email
MAX_PER_AUTHOR = 2        # at most this many main papers per trusted author
CITED_MAX_IDS = 40        # papers linked by trusted sources that we look up directly
CITED_MAX_AGE_MONTHS = 2  # ... only if their arXiv id is this recent
CITED_WINDOW_DAYS = 45
CITED_BASE_SCORE = 8      # ranking base for an off-topic paper a trusted source links to
TOPICS = {
    "model psychology": ["sycophancy", "sycophantic", "persona", "introspection", "deception", "deceptive",
                         "honesty", "emergent misalignment", "alignment faking", "scheming", "subliminal",
                         "personality", "self-report", "trait", "role-play", "persuasion", "manipulat",
                         "character", "lie detect"],
    "interpretability": ["interpretab", "mechanistic", "sparse autoencoder", "probe", "probing", "steering",
                         "representation", "circuit", "attribution", "whitebox", "activation"],
    "oversight & control": ["oversight", "monitor", "control", "sabotage", "untrusted", "chain-of-thought",
                            "chain of thought", "audit", "collusion", "latent reasoning"],
    "evals & capabilities": ["benchmark", "evaluation", "red-team", "red team", "dangerous capabilit",
                             "propensit", "uplift", "autonomy", "incident", "capabilit"],
    "training methods": ["rlhf", "reward", "preference", "post-training", "midtraining", "fine-tun",
                         "distillation", "reinforcement learning", "constitutional", "alignment training",
                         "pretraining", "rlvr", "dpo", "on-policy"],
    "robustness & security": ["jailbreak", "adversarial", "backdoor", "poison", "attack", "robust",
                              "prompt injection", "exploit", "vulnerab", "security"],
    "governance & risk": ["policy", "governance", "regulat", "risk", "standard", "safety case", "assessment",
                          "forecast", "responsible scaling", "frontier safety", "welfare"],
    "agents": ["agent", "multi-agent", "memory", "tool use", "long-horizon", "autonomous"],
}
# Selection: first the single most important item on each topic (if it clears a quality
# bar), then the rest purely by importance, so several strong items on one topic can
# all appear. Minimums keep blogs and papers balanced.
GENERAL_RESERVED = 1      # this many General-AI slots are guaranteed (the best general item(s) each week)
COVERAGE_SLOTS = 6        # at most this many "best on its topic" picks
COVERAGE_MIN_RANK = 14    # ... and only if that item's rank is at least this
MIN_POSTS = 3             # keep at least this many blog posts (if there are enough)
MIN_PAPERS = 3            # ... and this many approved papers
POSTS_MAX = 6
PAPERS_MAX = 6
TOTAL_ITEMS = 10          # hard cap on everything in one email (posts + papers + general)
GENERAL_SHOWN = 2         # "General AI worth knowing": popular/buzzing papers, notable training papers, major-lab posts
# "Lots of people are talking about this": Hacker News stories about AI with at least
# this many points in the last HN_DAYS days are treated as buzz (no names hardcoded).
HN_DAYS = 10
HN_MIN_POINTS = 100
HN_MAX_ITEMS = 1          # at most this many standalone Hacker News stories per email
AI_DOMAINS = ("arxiv.org", "huggingface.co", "openai.com", "anthropic.com", "deepmind.google",
              "deepmind.com", "sakana.ai", "mistral.ai", "x.ai", "ai.meta.com", "research.google",
              "lesswrong.com", "alignmentforum.org", "transformer-circuits.pub", "thinkingmachines.ai")
AI_TITLE_RE = re.compile(
    r"\b(AI|A\.I\.|LLMs?|GPT-?\d*|ChatGPT|Claude|Gemini|OpenAI|Anthropic|DeepMind|DeepSeek|"
    r"Mistral|Llama|Qwen|Grok|neural networks?|language models?|machine learning|deep learning|"
    r"reinforcement learning|foundation models?|reasoning models?|open[- ]weights?|agentic|RLHF|"
    r"fine-tuning|self-improvement|chatbots?|sycophancy|jailbreaks?|alignment faking)\b", re.I)
# Feeds whose text is only a boilerplate blurb: read the article itself (max N per run).
THIN_SOURCES = {"Anthropic Research": 12, "Anthropic Frontier Red Team": 6,
                "Google DeepMind": 8, "OpenAI Research": 8}
BOILERPLATE = ("Anthropic is an AI safety and research company",)
MAX_PER_SOURCE = 2   # so one blog can't fill the posts section
MAX_AGE_HOURS = 30 * 24   # candidate window
RECENT_DAYS, RECENT_BONUS = 7, 3    # ranking bonus: published in the last week
MID_DAYS, MID_BONUS = 14, 1         # ... or the last two weeks
MIN_RELEVANCE = 3         # needed (together with a core term) to be eligible
AUTHOR_BONUS = 8          # per trusted author on a paper (max 2 counted)
UPVOTE_STEP = 10          # +1 ranking point per this many HF upvotes ...
UPVOTE_BONUS_MAX = 6      # ... up to this many, and only if the paper is on-topic
REL_CAP = 12              # keyword relevance counts for at most this much when ranking
HF_APPROVAL_UPVOTES = 10  # HF upvotes needed to count as outside approval
ENDORSE_BONUS = 4         # ranking points per trusted source that links a paper (max 2)
ENDORSE_MIN_BONUS = 3     # sources with at least this bonus count as trusted endorsers
ON_TOPIC_REL = 6          # relevance needed for HF upvotes to count as approval (or a core term in the title)
GENERAL_MIN_UPVOTES = 100 # popular papers on ANY AI topic qualify for "General AI"
# Major AI blogs/newsletters whose non-safety posts can appear under "General AI".
GENERAL_SOURCES = {"Interconnects", "Transformer", "Anthropic Research", "Google DeepMind",
                   "OpenAI Research", "Hugging Face blog", "Raschka"}
UNVETTED_SHOWN = 1   # papers shown without any outside approval ...
UNVETTED_MIN_REL = 10     # ... but only when their keyword match is very strong
TRAINING_MIN_UPVOTES = 100 # popular training-technique papers get the small slot
TRAINING_MIN_SCORE = 3
HF_LOOKBACK_DAYS = 30
PAGE_MAX_NEW = 15         # newest N links on a no-feed index page are considered
ARXIV_PAGE = 100          # results per arXiv API request
ARXIV_MAX_PAGES = 3       # pages per keyword query (300 newest matches)
ARXIV_GAP = 10            # seconds between arXiv API requests (it rate-limits shared IPs)
ARXIV_BUDGET_SECONDS = 900  # give up on arXiv after this long in total
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
    # Import AI and Zvi (substack.com feeds) return HTTP 403 to GitHub's servers.
    ("Transformer", "https://www.transformernews.ai/feed", True, 4),
    ("Interconnects", "https://www.interconnects.ai/feed", True, 4),
    ("Raschka", "https://magazine.sebastianraschka.com/feed", False, 3),   # bonus 3: his links count as endorsements
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

# Low-volume, high-signal sources look back further than the 30-day default so
# a quiet month doesn't hide their best recent posts.
SOURCE_WINDOW_DAYS = {
    "Anthropic Alignment Science": 150,
    "Transformer Circuits": 120,
    "Redwood Research": 90,
    "METR": 90,
    "LessWrong (curated)": 60,
}


def cutoff_for(name, now):
    return now - timedelta(days=SOURCE_WINDOW_DAYS.get(name, MAX_AGE_HOURS / 24))

# Model psychology / behaviour. Title hit = 6, abstract/body hit = 3.
PRIORITY_KEYWORDS = [
    "sycophancy", "sycophantic", "persona", "deception", "deceptive", "alignment faking",
    "jailbreak", "jailbreaking", "sandbagging", "introspection", "emergent misalignment",
    "agentic misalignment", "reward hacking", "scheming", "self-preservation",
    "situational awareness", "evaluation awareness", "model psychology", "model character",
    "character training", "honesty", "persuasion", "role-play", "roleplay", "unfaithful",
    "backdoor", "sleeper agent", "model organism", "lie detector", "lie detection",
    "manipulative", "user manipulation", "assistant character", "assistant persona",
    "persona trait",
]
# Core safety terms. Title hit = 4, abstract/body hit = 2.
SAFETY_KEYWORDS = [
    "ai safety", "misalignment", "interpretability", "mechanistic interpretability",
    "sparse autoencoder", "activation steering", "red teaming", "red-teaming",
    "ai control", "oversight", "scalable oversight", "dangerous capabilities",
    "monitorability", "cot monitoring", "chain-of-thought monitoring", "ai monitor",
    "cot faithfulness", "chain-of-thought faithfulness", "safety training",
    "constitutional ai", "model spec", "alignment research", "alignment training",
    "ai governance", "ai policy", "safety case", "responsible scaling", "frontier safety",
    "model welfare", "prompt injection", "data poisoning", "adversarial robustness",
    "specification gaming", "goal misgeneralization", "power-seeking", "corrigibility",
    "agent foundations", "singular learning theory", "eliciting latent knowledge",
    "weak-to-strong", "multi-agent safety", "dangerous capability",
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
    t = re.sub(r"\s+[-|\u2013\u2014]\s+[^-|\u2013\u2014]{0,40}\bBlog\s*$", "", t, flags=re.I)
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
LLM_RE = re.compile(r"\b(language models?|LLMs?|reasoning models?|chatbots?|AI assistants?)\b", re.I)


def norm_name(n):
    n = n.lower().replace("-", " ")
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


TRUSTED_NORM = {norm_name(a) for a in TRUSTED_AUTHORS}


TOPIC_RE = {t: [re.compile(r"\b" + re.escape(k), re.I) for k in kws] for t, kws in TOPICS.items()}


def topic_of(it):
    """Crude topic bucket (used only to keep the email varied)."""
    title, text = it.get("title", ""), it.get("summary", "")
    best, best_n = "general AI", 0
    for topic, rxs in TOPIC_RE.items():
        n = sum(2 for rx in rxs if rx.search(title)) + sum(1 for rx in rxs if rx.search(text))
        if n > best_n:
            best, best_n = topic, n
    return best


def trusted_in(entry):
    names = [a.get("name", "") for a in entry.get("authors", [])]
    return [n for n in names if norm_name(n) in TRUSTED_NORM]


def _hits(rxs, kws, title, text, title_w, body_w):
    out = []
    for rx, kw in zip(rxs, kws):
        if rx.search(title):
            out.append([kw, title_w])
        elif rx.search(text):
            out.append([kw, body_w])
    return out


def _merge_overlaps(hits):
    """'interpretability' inside 'mechanistic interpretability' is one idea: count it once."""
    for h in hits:
        for o in hits:
            if o is h or h[1] is None or o[1] is None or o[0] == h[0]:
                continue
            if re.search(r"\b" + re.escape(h[0]) + r"s?\b", o[0]):   # h is part of o
                o[1] = max(o[1], h[1])
                h[1] = None
                break
    return [h for h in hits if h[1] is not None]


def score(title, text):
    """-> (relevance, has_core_term, training_score)."""
    hits = _merge_overlaps(
        _hits(PRIORITY_RE, PRIORITY_KEYWORDS, title, text, 6, 3)
        + _hits(SAFETY_RE, SAFETY_KEYWORDS, title, text, 4, 2))
    rel, strong, train = sum(w for _, w in hits), bool(hits), 0
    for rx in TRAINING_RE:
        if rx.search(title):
            train += 2
        elif rx.search(text):
            train += 1
    if train and not LLM_RE.search(f"{title} {text}"):
        train = 0                       # training-technique credit only for LLM work
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


def page_date(ap, raw_html, index_nodes, url, index_month=""):
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
    d = parse_date_str(index_month)             # month heading above the link on the index
    if d:
        return d
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
        self._month, self.anchor_month = "", {}  # "July 2026" heading above each link

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
            self._href = a["href"]
            self.anchors.setdefault(self._href, [])
            self.anchor_month.setdefault(self._href, self._month)
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
            elif re.fullmatch(rf"{_MON_RE}\s+\d{{4}}", text, re.I):
                self._month = text
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


def arxiv_author_urls(chunk=30):
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


STATS = {}   # source -> counts for the "source check" footer


def stat(source, key, n=1):
    d = STATS.setdefault(source, {"fetched": 0, "window": 0, "eligible": 0})
    d[key] += n


BUZZ = {}          # normalised url -> {"points", "comments", "url"} for AI stories on HN
BUZZ_PAPERS = {}   # arXiv id -> same, for papers linked from HN
HF_LINK_RE = re.compile(r"huggingface\.co/papers/(\d{4}\.\d{4,5})", re.I)
ARXIV_LINK_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
ENDORSEMENTS = {}   # arXiv id -> set of trusted source names that link to it


def note_endorsements(source, bonus, *texts):
    if bonus < ENDORSE_MIN_BONUS:
        return
    for t in texts:
        for pid in ARXIV_LINK_RE.findall(t or ""):
            ENDORSEMENTS.setdefault(pid, set()).add(source)


def apply_endorsements(items):
    for it in items:
        if it.get("kind") != "paper" or not it["id"].startswith("arxiv:"):
            continue
        srcs = sorted(ENDORSEMENTS.get(it["id"][6:], ()))
        if srcs:
            it["score"] += ENDORSE_BONUS * min(len(srcs), 2)
            it["approvals"] = list(it.get("approvals", [])) + [f"cited: {x}" for x in srcs[:2]]


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


FETCHES = {}   # per-run counter for richer_text()


def richer_text(name, link, raw):
    """Some feeds give only a boilerplate blurb. For a few sources, read the article."""
    plain = clean(raw, 400)
    boiler = plain.startswith(BOILERPLATE)
    cap = THIN_SOURCES.get(name)
    if not cap or (len(plain) >= 80 and not boiler):
        return "" if boiler else raw
    if FETCHES.get(name, 0) >= cap or not (link or "").startswith("http"):
        return "" if boiler else raw
    FETCHES[name] = FETCHES.get(name, 0) + 1
    try:
        art, _ = http_get(link)
    except Exception:
        return "" if boiler else raw
    time.sleep(1)
    ap = _LinkMeta()
    ap.feed(art)
    meta = ap.meta.get("og:description") or ap.meta.get("description") or ""
    if clean(meta).startswith(BOILERPLATE):
        meta = ""
    body = " ".join(p for p in ap.paragraphs if len(p) >= 60)[:1500]
    text = f"{meta} {body}".strip()
    return text or ("" if boiler else raw)


def entries_to_items(entries, name, always, bonus, is_paper, seen, cutoff):
    items = []
    for e in entries:
        uid = e.get("id") or e.get("link")
        if is_paper and uid:
            m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", uid)
            if m:
                uid = "arxiv:" + m.group(1)  # same id as the Hugging Face source
        stat(name, "fetched")
        when = entry_time(e)
        if when and when < cutoff:
            continue
        stat(name, "window")
        raw = e.get("summary") or e.get("description")
        if not is_paper:      # note which arXiv papers trusted sources link to
            blob = (raw or "") + " ".join(c.get("value", "") for c in (e.get("content") or []))
            note_endorsements(name, bonus, blob)
        if not uid or uid in seen:
            continue
        if not is_paper:
            raw = richer_text(name, e.get("link", ""), raw)
        title = clean_title(clean(e.get("title"), 200), e.get("link", ""))
        # Some feeds (LessWrong) ship the whole post: score the opening ~1500
        # chars, but only show 320.
        rel, strong, train = score(title, clean(raw, 1500))
        trusted = trusted_in(e) if is_paper else []
        ok, training_only = judge(rel, strong, train, always, bool(trusted))
        general = False
        if name in GENERAL_SOURCES and not is_paper and not (strong and rel >= MIN_RELEVANCE):
            ok, general = True, True     # not safety-specific: "General AI" candidate
        if not ok:
            continue
        extra = AUTHOR_BONUS * min(len(trusted), 2) if is_paper else bonus
        stat(name, "eligible")
        items.append({
            "id": uid, "source": name, "title": title, "summary": clean(raw, 320),
            "link": e.get("link", uid), "score": min(rel, REL_CAP) + min(train, 3) + extra,
            "rel": rel, "strong": strong, "training_only": training_only, "general": general,
            "trusted": trusted, "approvals": ["author"] if trusted else [],
            "upvotes": None, "date": fmt_date(when), "when": when,
            "kind": "paper" if is_paper else "post",
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


def arxiv_parse(url, deadline):
    """Fetch one arXiv API page, backing off on 429/503/timeouts.
    -> (feed, "") on success (possibly zero entries), (None, diagnosis) on failure."""
    waits, diag = [20, 45, 90], ""
    for attempt in range(len(waits) + 1):
        wait = None
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return feedparser.parse(r.read()), ""
        except HTTPError as exc:
            diag = f"HTTP {exc.code}"
            ra = exc.headers.get("Retry-After") if exc.headers else None
            wait = int(ra) if ra and str(ra).isdigit() else None
        except Exception as exc:
            diag = f"{type(exc).__name__}: {str(exc)[:80]}"
        if attempt == len(waits) or time.time() > deadline:
            break
        time.sleep(min(wait or waits[attempt], 120))
    return None, diag


def fetch_arxiv(name, url, seen, cutoff, errors, max_pages, deadline):
    """Page back through an arXiv API query until results pass the cutoff.
    Returns (items, ok)."""
    items = []
    for page in range(max_pages):
        if page:
            time.sleep(ARXIV_GAP)
        feed, diag = arxiv_parse(f"{url}&start={page * ARXIV_PAGE}", deadline)
        if feed is None:
            errors.append(f"{name}: failed after retries ({diag}) - {url[:100]}")
            return items, page > 0           # earlier pages still count
        if not feed.entries:
            break                            # an empty result is not an error
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
    stat(name, "fetched", len(found))
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
        note_endorsements(name, bonus, art)
        title = clean_title(clean(ap.meta.get("og:title") or ap.title, 200), full)
        nodes = p.anchors.get(href, [])
        date = page_date(ap, art, nodes, full, p.anchor_month.get(href, ""))
        when = display_to_datetime(date)
        if cutoff and when and when < cutoff:
            continue
        stat(name, "window")
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
        stat(name, "eligible")
        items.append({
            "id": nid, "source": name, "title": title, "summary": clean(desc, 320),
            "link": full, "score": min(rel, REL_CAP) + min(train, 3) + bonus, "rel": rel,
            "approvals": [], "strong": strong,
            "training_only": training_only, "trusted": [], "upvotes": None,
            "date": date, "when": when, "kind": "post",
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
        stat("Hugging Face daily papers", "fetched")
        when = parse_iso(p.get("submittedOnDailyAt") or row.get("publishedAt"))
        if when and when < cutoff:
            continue
        stat("Hugging Face daily papers", "window")
        title = clean_title(clean(p.get("title") or row.get("title"), 200))
        abstract = clean(p.get("summary") or row.get("summary"), 1500)
        upv = p.get("upvotes") or 0
        trusted = [a["name"] for a in p.get("authors", [])
                   if norm_name(a.get("name", "")) in TRUSTED_NORM]
        rel, strong, train = score(title, abstract)
        ok, training_only = judge(rel, strong, train, False, bool(trusted), upv)
        # Upvotes only count as approval for papers that are really on-topic:
        # a core term in the title, or a strong overall keyword match.
        topical = rel >= ON_TOPIC_REL or score(title, "")[1]
        general = False
        if upv >= GENERAL_MIN_UPVOTES and not topical and not training_only and not trusted:
            ok, general = True, True        # popular, but not safety-specific
        if not ok:
            continue
        bonus = AUTHOR_BONUS * min(len(trusted), 2)
        if topical:
            bonus += min(upv // UPVOTE_STEP, UPVOTE_BONUS_MAX)
        approvals = (["author"] if trusted else []) + \
                    (["hf"] if upv >= HF_APPROVAL_UPVOTES and topical else [])
        stat("Hugging Face daily papers", "eligible")
        total = min(upv // 20, 15) if general else min(rel, REL_CAP) + min(train, 3) + bonus
        items.append({
            "id": f"arxiv:{pid}", "source": "Hugging Face daily papers",
            "title": title, "summary": clean(abstract, 320),
            "link": f"https://arxiv.org/abs/{pid}", "score": total, "rel": rel,
            "strong": strong, "training_only": training_only, "general": general,
            "trusted": trusted, "approvals": approvals, "upvotes": upv, "when": when,
            "kind": "paper",
            "date": fmt_date(parse_iso(row.get("publishedAt")) or when),
        })
    return items


# -------------------------------------------------------------- selection ---
def fetch_cited_papers(items, seen, now, errors, deadline):
    """Look up recent arXiv papers that trusted sources link to but that our own
    queries didn't fetch. On-topic ones become ordinary papers (approved via the
    citation); everything else becomes a "General AI" candidate. This is how an
    important paper outside the safety keywords (e.g. a new training paradigm)
    reaches you when Interconnects, Transformer, Redwood, LessWrong etc. discuss it."""
    have = {i["id"] for i in items}
    idx_now = (now.year % 100) * 12 + now.month
    ids = []
    for pid in sorted(set(ENDORSEMENTS) | set(BUZZ_PAPERS), reverse=True):   # newest ids first
        if f"arxiv:{pid}" in seen or f"arxiv:{pid}" in have:
            continue
        idx = int(pid[:2]) * 12 + int(pid[2:4])
        if 0 <= idx_now - idx <= CITED_MAX_AGE_MONTHS:
            ids.append(pid)
    ids = ids[:CITED_MAX_IDS]
    out, cutoff, src = [], now - timedelta(days=CITED_WINDOW_DAYS), "arXiv (cited by trusted sources)"
    for i in range(0, len(ids), 20):
        if i:
            time.sleep(ARXIV_GAP)
        url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids[i:i + 20]) + "&max_results=20"
        feed, diag = arxiv_parse(url, deadline)
        if feed is None:
            errors.append(f"{src}: failed ({diag})")
            break
        for e in feed.entries:
            m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", e.get("id") or "")
            if not m:
                continue                                          # arXiv error entries
            stat(src, "fetched")
            when = entry_time(e)
            if when and when < cutoff:
                continue
            stat(src, "window")
            title = clean_title(clean(e.get("title"), 200), e.get("link", ""))
            raw = e.get("summary")
            rel, strong, train = score(title, clean(raw, 1500))
            trusted = trusted_in(e)
            ok, _ = judge(rel, strong, train, False, bool(trusted))
            topical = ok and (rel >= ON_TOPIC_REL or score(title, "")[1])
            stat(src, "eligible")
            out.append({
                "id": "arxiv:" + m.group(1), "source": src, "title": title,
                "summary": clean(raw, 320), "link": e.get("link", ""), "rel": rel,
                "score": (min(rel, REL_CAP) + AUTHOR_BONUS * min(len(trusted), 2)) if topical
                         else CITED_BASE_SCORE,
                "strong": strong, "training_only": False, "general": not topical,
                "trusted": trusted, "approvals": ["author"] if trusted else [],
                "upvotes": None, "date": fmt_date(when), "when": when, "kind": "paper",
            })
    return out


def fetch_hn(seen, now, errors):
    """Hacker News stories about AI with lots of points: a direct measure of
    'lots of people are talking about this'. Papers linked from HN feed
    BUZZ_PAPERS; other stories become General-AI candidates (or ordinary posts
    if they match the safety keywords)."""
    ts = int((now - timedelta(days=HN_DAYS)).timestamp())
    url = ("https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=500"
           f"&numericFilters=created_at_i%3E{ts},points%3E{HN_MIN_POINTS - 1}")
    try:
        data = json.loads(http_get(url)[0])
    except Exception as exc:
        errors.append(f"Hacker News: {exc}")
        return []
    src, items = "Hacker News", []
    for h in data.get("hits", []):
        stat(src, "fetched")
        title = clean_title(h.get("title") or "")
        link = h.get("url") or ""
        host = urlparse(link).netloc.lower().removeprefix("www.")
        if not (AI_TITLE_RE.search(title)
                or any(host == d or host.endswith("." + d) for d in AI_DOMAINS)):
            continue
        stat(src, "window")
        pts, com = h.get("points") or 0, h.get("num_comments") or 0
        hn_link = f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        buzz = {"points": pts, "comments": com, "url": hn_link}
        m = ARXIV_LINK_RE.search(link) or HF_LINK_RE.search(link)
        if m:
            BUZZ_PAPERS[m.group(1)] = buzz
            continue
        if link:
            BUZZ[norm_url(link)] = buzz
        uid = f"hn:{h.get('objectID')}"
        if uid in seen:
            continue
        rel, strong, train = score(title, "")
        general = True            # HN stories never take a main blog/paper slot
        when = datetime.fromtimestamp(h.get("created_at_i") or 0, timezone.utc)
        stat(src, "eligible")
        items.append({
            "id": uid, "source": src, "title": title, "summary": "", "link": link or hn_link,
            "score": min(pts // 100, 5) + min(com // 200, 2),
            "rel": rel, "strong": strong, "training_only": False, "general": general,
            "trusted": [], "approvals": [], "upvotes": None, "buzz": buzz,
            "date": fmt_date(when), "when": when, "kind": "post",
        })
    return items


def apply_buzz(items):
    """Extra weight for anything many people are discussing on Hacker News, and
    drop an HN story when we already have the same article from a feed."""
    have = {norm_url(i.get("link", "")) for i in items if i["source"] != "Hacker News"}
    out = []
    for it in items:
        if it["source"] == "Hacker News":
            if norm_url(it["link"]) in have:
                continue                                   # same article already present
            out.append(it)
            continue
        b = BUZZ.get(norm_url(it.get("link", ""))) or \
            (BUZZ_PAPERS.get(it["id"][6:]) if it["id"].startswith("arxiv:") else None)
        if b:
            it["buzz"] = b
            it["score"] += min(b["points"] // 100, 4)
            if it.get("kind") == "paper":
                it["approvals"] = list(it.get("approvals", [])) + ["hn"]
        out.append(it)
    return out


def merge(items):
    """Same paper/post can arrive via several feeds or queries: keep the
    best-scored copy but pool the approvals, trusted authors and upvotes."""
    best = {}
    for it in items:
        cur = best.get(it["id"])
        if cur is None:
            best[it["id"]] = it
            continue
        win, lose = (it, cur) if it["score"] > cur["score"] else (cur, it)
        m = dict(win)
        m["approvals"] = sorted(set(win.get("approvals", [])) | set(lose.get("approvals", [])))
        m["trusted"] = list(dict.fromkeys(list(win["trusted"]) + list(lose["trusted"])))
        ups = [u for u in (win.get("upvotes"), lose.get("upvotes")) if u]
        m["upvotes"] = max(ups) if ups else None
        m["general"] = bool(win.get("general") and lose.get("general"))
        m["buzz"] = win.get("buzz") or lose.get("buzz")
        best[it["id"]] = m
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


def recency_bonus(when, now, kind="paper"):
    if not when:
        return 0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = now - when
    if kind != "paper":                       # posts: recency matters more
        for days, bonus in POST_RECENT:
            if age <= timedelta(days=days):
                return bonus
        return 0
    if age <= timedelta(days=RECENT_DAYS):
        return RECENT_BONUS
    if age <= timedelta(days=MID_DAYS):
        return MID_BONUS
    return 0


def take_posts(items, n, per_source, now):
    """Like take(), but at most POSTS_OLD_MAX posts older than POSTS_OLD_DAYS."""
    out, counts, old = [], {}, 0
    for it in items:
        if len(out) >= n:
            break
        if counts.get(it["source"], 0) >= per_source:
            continue
        w = it.get("when")
        if w and w.tzinfo is None:
            w = w.replace(tzinfo=timezone.utc)
        is_old = bool(w) and (now - w) > timedelta(days=POSTS_OLD_DAYS)
        if is_old and old >= POSTS_OLD_MAX:
            continue
        old += is_old
        counts[it["source"]] = counts.get(it["source"], 0) + 1
        out.append(it)
    return out


def take_authors(items, per_author):
    """No more than per_author papers by the same trusted author."""
    out, used = [], {}
    for it in items:
        if any(used.get(a, 0) >= per_author for a in it["trusted"]):
            continue
        for a in it["trusted"]:
            used[a] = used.get(a, 0) + 1
        out.append(it)
    return out


def select(items, now=None):
    now = now or datetime.now(timezone.utc)
    pool = merge(items)
    for it in pool:
        it["rank"] = it["score"] + recency_bonus(it.get("when"), now, it.get("kind"))
        it["topic"] = topic_of(it)
    pool.sort(key=lambda x: x["rank"], reverse=True)
    paper = lambda i: i.get("kind") == "paper"
    live = [i for i in pool if not i["training_only"] and not i.get("general")]
    gen, n_hn = [], 0
    for i in pool:
        if i.get("general") or i["training_only"]:
            if i["source"] == "Hacker News":
                n_hn += 1
                if n_hn > HN_MAX_ITEMS:
                    continue
            gen.append(i)

    # Candidates (best first). Per-source / per-author / old-post limits are applied
    # here, in rank order, so any subset of these lists respects them.
    posts_c = take_posts([i for i in live if not paper(i)], 999, MAX_PER_SOURCE, now)
    papers_c = take_authors([i for i in live if paper(i) and i.get("approvals")], MAX_PER_AUTHOR)
    # Papers without outside approval go in the small, clearly labelled "unvetted" section.
    unvetted_c = [i for i in live if paper(i) and not i.get("approvals")
                  and i.get("rel", 0) >= UNVETTED_MIN_REL]
    section = {}
    for i in posts_c:
        section[i["id"]] = "posts"
    for i in papers_c:
        section[i["id"]] = "papers"
    for i in gen:
        section[i["id"]] = "general"
    core = sorted(posts_c + papers_c + gen, key=lambda x: x["rank"], reverse=True)
    caps = {"posts": POSTS_MAX, "papers": PAPERS_MAX, "general": GENERAL_SHOWN}

    chosen = {"posts": [], "papers": [], "general": [], "unvetted": []}
    picked = set()

    def total():
        return sum(len(v) for v in chosen.values())

    def add(it, sec):
        if it["id"] in picked:                  # never include the same item twice
            return
        chosen[sec].append(it)
        picked.add(it["id"])

    if unvetted_c and UNVETTED_SHOWN:                     # reserved slot(s)
        for it in unvetted_c[:UNVETTED_SHOWN]:
            add(it, "unvetted")

    for it in [i for i in core if section[i["id"]] == "general"][:GENERAL_RESERVED]:
        if total() < TOTAL_ITEMS and it["id"] not in picked:
            add(it, "general")                                # reserved: the best general item

    # 1. Coverage: the most important item on each topic (not merely the first one
    #    seen), strongest topics first, only if it is important enough.
    best = {}
    for it in core:                                      # core is rank-sorted
        if it["rank"] >= COVERAGE_MIN_RANK:
            best.setdefault(it["topic"], it)
    for it in sorted(best.values(), key=lambda x: x["rank"], reverse=True)[:COVERAGE_SLOTS]:
        sec = section[it["id"]]
        if total() < TOTAL_ITEMS and len(chosen[sec]) < caps[sec] and it["id"] not in picked:
            add(it, sec)

    # 2. Balance: keep blogs and papers both represented.
    for sec, floor in (("posts", MIN_POSTS), ("papers", MIN_PAPERS)):
        for it in core:
            if len(chosen[sec]) >= floor or total() >= TOTAL_ITEMS:
                break
            if section[it["id"]] == sec and it["id"] not in picked:
                add(it, sec)

    # 3. Everything else purely by importance, so several strong items on the same
    #    topic can all be included.
    for it in core:
        if total() >= TOTAL_ITEMS:
            break
        sec = section[it["id"]]
        if it["id"] not in picked and len(chosen[sec]) < caps[sec]:
            add(it, sec)

    for k in chosen:
        chosen[k].sort(key=lambda x: x["rank"], reverse=True)
    return chosen["posts"], chosen["papers"], chosen["unvetted"], chosen["general"], [], pool


# ----------------------------------------------------------------- output ---
def render(posts, papers, unvetted, general, training, errors, stats=None):
    today = datetime.now(timezone.utc).strftime("%a %d %b %Y")

    def item_html(it, n):
        badge = ""
        if it["trusted"]:
            badge = (" &middot; <span style='color:#b8860b'>&#9733; "
                     + html.escape(", ".join(it["trusted"][:3])) + "</span>")
        if it.get("upvotes"):
            badge += f" &middot; <span style='color:#555'>&#9650; {it['upvotes']}</span>"
        cited = [a[7:] for a in it.get("approvals", []) if a.startswith("cited: ")]
        if cited:
            badge += (" &middot; <span style='color:#2a6'>cited by "
                      + html.escape(", ".join(cited)) + "</span>")
        if it.get("buzz"):
            b = it["buzz"]
            badge += (f" &middot; <a href='{html.escape(b['url'])}' style='color:#c60;"
                      f"text-decoration:none'>Hacker News &#9650; {b['points']} "
                      f"&middot; {b['comments']} comments</a>")
        date_html = (f" &middot; {html.escape(it['date'])}" if it.get("date")
                     else " &middot; <span style='color:#999'>date unknown</span>")
        topic_html = f" &middot; {html.escape(it['topic'])}" if it.get("topic") else ""
        return (
            "<div style='margin:0 0 14px'>"
            f"<a href='{html.escape(it['link'])}' style='font-weight:600;"
            f"text-decoration:none'>{html.escape(it['title'])}</a>"
            f"<div style='color:#666;font-size:12px'>{html.escape(it['source'])}{date_html}{topic_html}{badge}</div>"
            + (f"<div style='font-size:14px;margin-top:2px'>{html.escape(clean(it['summary'], n))}</div>"
               if it["summary"] else "") +
            "</div>"
        )

    def block(heading, items, n, note=""):
        if not items:
            return ""
        note_html = f"<div style='color:#888;font-size:12px;margin:-4px 0 10px'>{note}</div>" if note else ""
        return (f"<h3 style='margin:26px 0 8px'>{heading}</h3>" + note_html
                + "".join(item_html(i, n) for i in items))

    body = (block("Blogs &amp; newsletters", posts, 300)
            + block("Papers", papers, 300)
            + block("Unvetted papers", unvetted, 240,
                    "Strong keyword match, but no trusted author, Hugging Face upvotes, "
                    "or link from a trusted source (yet).")
            + block("General AI worth knowing", general, 240,
                    "Not safety-specific: picked for popularity (100+ Hugging Face upvotes) "
                    "or Hacker News buzz, or because it's from a major AI blog or newsletter, or a notable training paper.")
            + block("Training &amp; methods", training, 180))
    if not body:
        body = "<p>Nothing new matched this week.</p>"
    err = ""
    if errors:
        err = ("<hr><p style='color:#a00;font-size:12px'>Feed problems:<br>"
               + "<br>".join(html.escape(x) for x in errors) + "</p>")
    stats_html = ""
    if stats:
        stats_html = ("<hr><div style='color:#888;font-size:11px'>Source check "
                      "(fetched &middot; in window &middot; eligible &middot; shown):<br>"
                      + "<br>".join(html.escape(x) for x in stats) + "</div>")
    page = (
        "<div style='font-family:-apple-system,Segoe UI,sans-serif;max-width:640px'>"
        f"<h2 style='margin:0'>Safety digest &middot; weekly</h2><div style='color:#666'>{today}</div>"
        f"{body}{err}{stats_html}</div>"
    )
    text_lines = [f"Safety digest (weekly) - {today}", ""]
    for heading, items in (("BLOGS & NEWSLETTERS", posts), ("PAPERS", papers),
                           ("UNVETTED PAPERS (no outside approval yet)", unvetted),
                           ("GENERAL AI WORTH KNOWING", general),
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
    ENDORSEMENTS.clear()
    STATS.clear()
    FETCHES.clear()
    BUZZ.clear()
    BUZZ_PAPERS.clear()
    errors, items = [], []

    for name, url, always, bonus in FEEDS:
        items += fetch(name, url, always, bonus, False, seen, cutoff_for(name, now), errors)
    for name, url, link_re, always, bonus in PAGES:
        items += fetch_page_source(name, url, link_re, always, bonus, seen, errors,
                                   cutoff_for(name, now))

    # Trusted-author queries first (fewest requests, most valuable), then keywords.
    deadline = time.time() + ARXIV_BUDGET_SECONDS
    jobs = [("arXiv (trusted authors)", u, 2) for u in arxiv_author_urls()] + \
           [("arXiv", u, ARXIV_MAX_PAGES) for u in arxiv_keyword_urls()]
    failed_in_a_row = 0
    for i, (name, url, pages) in enumerate(jobs):
        if time.time() > deadline:
            errors.append("arXiv: time budget used up, remaining queries skipped")
            break
        if i:
            time.sleep(ARXIV_GAP)
        got, ok = fetch_arxiv(name, url, seen, cutoff, errors, pages, deadline)
        items += got
        failed_in_a_row = 0 if ok else failed_in_a_row + 1
        if failed_in_a_row >= 3:
            errors.append("arXiv: three queries in a row failed, remaining skipped")
            break
    items += fetch_hf_papers(seen, now, errors)
    items += fetch_hn(seen, now, errors)
    if failed_in_a_row < 3 and time.time() < deadline:
        time.sleep(ARXIV_GAP)
        items += fetch_cited_papers(items, seen, now, errors, deadline)

    apply_endorsements(items)
    items = apply_buzz(items)
    posts, papers, unvetted, general, training, pool = select(items, now)
    chosen = posts + papers + unvetted + general + training
    enrich_descriptions(chosen)
    shown = {}
    for it in chosen:
        shown[it["source"]] = shown.get(it["source"], 0) + 1
    stats = [f"{n}: {c['fetched']} fetched · {c['window']} in window · {c['eligible']} eligible · "
             f"{shown.get(n, 0)} shown" for n, c in sorted(STATS.items())]
    if any(n == "Hacker News" for n in STATS):
        stats.append("Hacker News data: search by Algolia (hn.algolia.com)")
    page, text = render(posts, papers, unvetted, general, training, errors, stats)
    if dry:
        Path("digest.html").write_text(page)
        print(f"dry run: {len(posts)} posts, {len(papers)} papers, {len(unvetted)} unvetted, {len(general)} general, {len(training)} training, "
              f"{len(pool)} eligible in window, {len(errors)} feed errors")
        return

    send(page, text, len(chosen))

    # Only what was actually shown counts as seen; near-misses stay eligible
    # until they age out of their window and carry over to later weeks.
    for it in chosen:
        seen[it["id"]] = now.isoformat()
    keep_after = (now - timedelta(days=180)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= keep_after}
    SEEN_FILE.write_text(json.dumps(seen, indent=1))


if __name__ == "__main__":
    main()
