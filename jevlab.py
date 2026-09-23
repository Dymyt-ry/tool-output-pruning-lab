#!/usr/bin/env python
"""jevlab - per-call keep/drop decisions for Claude Code transcripts, decided by a local Laya.

Nothing here touches Claude Code. It reads session JSONL files read-only and
writes only inside this directory. Delete the directory to remove it.

    ./jevlab.py replay <session.jsonl> [--limit N]   decide, and compare to the age baseline
    ./jevlab.py harvest [--out pairs.jsonl]          weak-labelled keep/drop pairs for a fine-tune
    ./jevlab.py bench                                latency and VRAM of one decision
    ./jevlab.py selftest                             parser + baseline maths, no model

The state is one call, not the conversation: goal + tool + input + a preview of
that call's own result. That is what fits Laya's 1024-token window; the upstream
design sends the whole 25k-token transcript and Laya silently keeps only its
first ~800 tokens (laya/common.py, build_sequence: st = st[:room]).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MODEL = os.environ.get("JEVLAB_MODEL", "convaiinnovations/laya-typed-decisions")
SESSIONS = Path.home() / ".claude" / "projects"

# A result these tools produced can be fetched again by re-running the call, so
# dropping it costs a repeat at worst. Everything else (a command with side
# effects, a network read, a subagent run) is a one-shot observation: issue #25
# on the upstream repo is a deleted valuation that could never be recomputed.
REPRODUCIBLE = {"Read", "Glob", "Grep", "NotebookRead", "TodoWrite", "LS"}

HEAD_CHARS, TAIL_CHARS = 600, 200
PRESERVE = 6  # newest calls: never candidates, and never labelled (nothing follows them yet)

# Host-written user turns: command echoes, reminders, notifications. Upstream
# issue #70 measured these becoming 54% of retained "user" text, and the default
# goal being the /compact echo rather than the task.
HOST_TEXT = re.compile(r"<(command-name|local-command|system-reminder|task-notification)")

# Anything that reaches a training set is in the weights for good, so credentials
# are stripped on the way out of a transcript, not later. Upstream issue #64 asks
# for the same before their state goes to a third-party API.
SECRETS = [
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret|password|passwd|pwd|auth)\b\s*[:=]\s*[\"']?[\w.\-/+=]{8,}"),
    re.compile(r"(?i)\bbearer\s+[\w.\-/+=]{8,}"),  # header form, no separator
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|xox[abprs]-[A-Za-z0-9\-]{10,})"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*\S{8,}"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),  # addresses identify people, not just accounts
    # vendor prefixes: cheap to list, and each one is a key that carries its own
    # scope. Absent from this corpus today, which is the point of adding them now.
    re.compile(
        r"\b(sk_live_|rk_live_|sk_test_|SG\.[\w\-]{10}|AIza[\w\-]{10}|glpat-|npm_|dckr_pat_"
        r"|hf_[A-Za-z0-9]{10}|r8_[A-Za-z0-9]{10}|fal_[A-Za-z0-9]{10}|xai-[A-Za-z0-9]{10}"
        r"|sk-ant-|AGE-SECRET-KEY-|shpat_|shpss_)[\w\-]*"
    ),
    # secrets that never take the KEY=VALUE shape
    re.compile(r"\b\w+://[^/\s:@]+:[^/\s@]+@"),  # scheme://user:pass@host
    re.compile(r"(?i)\bcurl\b[^\n|;]*\s-{1,2}u(?:ser)?[= ]\s*\S+:\S+"),
    re.compile(r"(?i)\b(PGPASSWORD|MYSQL_PWD|REDIS_PASSWORD)\b\s*=\s*\S+"),
    re.compile(r"(?i)\bmysql\b[^\n|;]*\s-p\S+"),
    re.compile(r"\$(?:2[aby]|6|5|y)\$\d{2}\$[./A-Za-z0-9]{20,}"),  # bcrypt / crypt(3) hashes
    re.compile(r"\bssh-(?:rsa|ed25519|dss) [A-Za-z0-9+/]{20,}={0,2}(?: \S+)?"),  # public, still an identifier
    # a phone number is a person. International form, and the Czech mobile shape
    re.compile(r"(?<![\w+])(?:\+|00)\d{2,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)"),
    re.compile(r"(?<![\d.,])[67]\d{2} \d{3} \d{3}(?![\d.,])"),
]

# Identifiers that are not secret but are a key into something public or into
# one physical machine: a MAC address is a network card, a commit hash can be
# searched on GitHub and returns the repository and its author.
MAC = re.compile(r"(?i)(?<![\w:])[0-9a-f]{2}(?:[:-][0-9a-f]{2}){5}(?![\w:])")
IPV6 = re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])")


def _ipv6(m: re.Match) -> str:
    # "12:30:45" is a time and "[::-1]" a slice; an address has a hex letter or
    # a "::", and more than one group
    s = m.group(0)
    groups = [g for g in s.split(":") if g]
    looks = len(s) >= 6 and len(groups) >= 2 and ("::" in s or re.search(r"(?i)[a-f]", s))
    return "[IP]" if looks else s
# 7-64 hex chars with at least one digit and one letter: a commit, a blob, a
# container. The digit rules out words like "defaced"; the dash rule leaves this
# tool's own buckets (name-1a2b3c4d) to STALE_BUCKET.
HEX_ID = re.compile(r"(?<![\w-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}(?![\w-])")
# A random mixed-case id (an ElevenLabs voice, an API object) points at one
# account's object. Identifiers written by people have words in them, so a
# lowercase run of five letters marks a name, not an id.
RANDOM_ID = re.compile(r"(?<![\w-])(?=[A-Za-z0-9]*\d[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])(?![A-Za-z0-9]*[a-z]{5})[A-Za-z0-9]{16,}(?![\w-])")
# A dotless machine name from a hosting provider: vmi1234567 is a VPS name
# and appears in every shell prompt on it, with no dot for BARE_HOST to find.
PROVIDER_HOST = re.compile(r"(?i)\b(?:vmi|vps|srv|ip-|ec2-)\d[\d-]{4,}\b")


# Credentials are not the only thing a transcript carries: home paths name the
# user, private network addresses name a network, project directories name
# customers. None of that is secret in the credential sense; all of it identifies a person, a network and a
# customer, and a fact repeated in half the rows is exactly what no per-record
# privacy measure can protect later.
PUBLIC_HOSTS = {
    "github.com", "api.github.com", "raw.githubusercontent.com", "gist.github.com",
    "huggingface.co", "www.kaggle.com", "kaggle.com", "pypi.org", "files.pythonhosted.org",
    "registry.npmjs.org", "www.npmjs.com", "crates.io", "docs.anthropic.com", "claude.ai",
    "code.claude.com", "stackoverflow.com", "developer.mozilla.org", "archlinux.org",
    "aur.archlinux.org", "localhost", "127.0.0.1", "0.0.0.0", "pytorch.org", "docker.io",
    "claude.com", "docs.claude.com", "hub.docker.com", "docker.com", "anthropic.com",
}
HOME_PATH = re.compile(r"(?<![\w.-])(/(?:home|Users)/)(?!USER\b)[\w.-]+")
PROJECT_DIR = re.compile(
    r"(?i)(?<![\w.-])((?:/(?:home|Users)/USER/(?:Projects|work)|/var/www|/srv|/opt)/)([\w.-]+)"
)
IPV4 = re.compile(r"\b(?!127\.0\.0\.1|0\.0\.0\.0)(?:\d{1,3}\.){3}\d{1,3}\b")
URL_HOST = re.compile(r"\b(https?://)([\w.-]+)")
# A domain written in prose ("deployed to example-shop.com") never reaches URL_HOST,
# which needs a scheme. The denylist does not save it either: a client's domain
# is often spelled differently from the directory it is developed in. So match
# bare hostnames on a TLD list - conservative, because a careless one turns
# setup.sh and script.pl into hostnames.
# Country codes plus the gTLDs that are not also ordinary identifiers. Left out
# on purpose: app, at, email, group, team, works, media, design, live, store,
# link, page, info - each of them turns m.group(1), arr.at(0) or user.email into
# a hostname. A missed exotic TLD costs one leaked domain; a matched identifier
# mangles every Python file in the corpus, so this list stays conservative and
# the client domains that matter go in the denylist as well.
TLDS = (
    "com|cz|sk|eu|net|org|io|ai|dev|cloud|tech|xyz|me|tv|gg|shop|site|online"
    "|uk|de|fr|it|es|nl|be|ch|hu|ro|se|no|fi|dk|pt|ie|ua|ru|cn|jp|au|ca|br|mx|co"
)
# The newer gTLDs small business sites commonly use. Each one here is a word that is
# not also a common attribute or file extension (.sh .so .in .pl .rs, font.family,
# Alignment.center, tz.zone); that is the whole selection rule.
TLDS += (
    "|chat|social|graphics|studio|agency|digital|network|systems|solutions|space|website"
    "|world|today|club|biz|academy|photography|gallery|marketing|consulting|legal"
    "|clinic|dental|fitness|coffee|restaurant|cafe|pizza|wine|travel|tours|estate|homes"
    "|construction|energy|solar|autos|fashion|clothing|beauty|salon|yoga|garden|farm"
    "|bio|eco|wedding|school|courses|careers|film|radio|games|ninja|guru|expert|wiki"
    "|vip|lol|rocks|lt|lv|ee|si|hr|bg|gr|tr|il|kr|tw|hk|sg|nz|za|us|pw"
)
BARE_HOST = re.compile(r"(?i)(?<![\w.@-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:" + TLDS + r"))\b(?![.(])")


# A directory called "tools" or "research" names nothing; one called after a
# client names everything. Without a system word list, the generic ones are
# listed here rather than guessed, and a name not on this list is treated as
# identifying - the safe direction.
GENERIC_NAMES = {
    "tools", "skills", "research", "setup", "work", "data", "docs", "test", "tests",
    "source", "build", "dist", "html", "public", "assets", "scripts", "config", "node",
    "python", "learning", "shared", "share", "temp", "backup", "archive", "sandbox",
    "projects", "project", "code", "repos", "notes", "personal", "default", "common",
    "http", "https", "main", "master", "index", "admin", "user", "users", "home",
    # names that reach the list from a Host alias, a VPS login or a git address
    # but identify nobody; bucketing them only pseudonymises the English language
    "github", "root", "noreply",
    "claude", "docker", "script", "result", "pages", "curl",
    "blog", "apps", "accounts", "news", "mail", "email", "shop", "store", "site",
    "team", "help", "demo", "beta", "dashboard", "login", "auth", "account",
    "static", "media", "image", "images", "video", "search", "download", "support",
    "portal", "server", "client", "service", "status", "health", "events", "files",
}


# Prose cannot be redacted word by word. Delete the verb from "the customer was
# not billed for the second month" and the sentence still says it: what leaks is the
# meaning, not the token. So a topic is detected and the whole row goes. Losing
# a row costs one training example; keeping it can publish a price list.
# Not \b: it would let BILLING_PLAN.md or PITCH_DECK_NOTES.md through,
# because an underscore is a word character. A letter is what continues a word.
SENSITIVE = re.compile(
    r"(?i)(?<![^\W\d_])("
    r"faktur\w*|vyúčtov\w*|naúčtov\w*|účtenk\w*|účtován\w*|účetn\w*|idoklad\w*|fakturoid\w*|isdoc"
    r"|invoice\w*|billing|fakturace"
    r"|ičo|dič|dph|iban|swift|číslo\s+účtu|vat|daň\w*|daně|daňov\w*"
    r"|smlouv\w*|smluvn\w*|contract\w*|nda|gdpr|dpo|obchodní\s+podmínk\w*|zpracování\s+osobních"
    r"|osobní\w*\s+údaj\w*|rodn\w+\s+čísl\w*|narozen\w*|telefon\w*"
    r"|ceník\w*|cenov\w*|cen(?:a|u|y|ou|ě|ami|ách)|kč|czk"
    r"|\d+\s*(?:eur|usd)"
    r"|platb\w*|platebn\w*|předplatn\w*|tarif\w*|provize|mzd\w*|výplat\w*|odměn\w*"
    r"|kredit\w*|credits?|subscription\w*|refund\w*|payout|pricing|revenue|salary|payroll"
    r"|stripe\w*|paypal\w*|gopay\w*|comgate\w*"
    # business context: customers, offers, orders, sales, plans, competitors
    r"|klient\w*|zákazní\w*|zákazník\w*|nabídk\w*|nabídn\w*|poptáv\w*|objednáv\w*"
    r"|obchodn\w*|prodej\w*|tržb\w*|marž\w*|zisk\w*|rozpočt\w*|rozpočet"
    r"|investor\w*|investic\w*|strategi\w*|roadmap\w*|konkurenc\w*|competitor\w*"
    r"|pitch[\s_-]*deck\w*|yc|y[\s_-]*combinator|fundrais\w*|valuation\w*|konverzní\w*"
    r")(?![^\W\d_])"
)
# Price shapes the word list cannot see: a currency sign next to a number. A
# bare $1 is a shell argument, so the dollar needs two digits or cents.
CURRENCY = re.compile(r"[€£]\s?\d|\d\s?[€£]|(?<![\w$])\$\d{2,}(?:[.,]\d+)?(?![\w$])|\$\d+\.\d{2}\b|\d,-(?!\w)")

# Whole outputs whose every line is a record about someone or something real.
# A pseudonym inside them does not help: a public leaderboard row with a rank
# and a score is a primary key into a public database, whatever the name says,
# and a commit log is a list of hashes that GitHub search turns into authors.
DROP_ROWS = {
    "leaderboard": re.compile(r"(?i)\bleaderboard\b|\bkaggle\s+c(?:ompetitions)?\s+(?:leaderboard|submissions)"),
    "whois": re.compile(r"(?i)\bwhois\b|^\s*(?:registrar|registrant|admin-c|nserver)\s*:", re.M),
    "git history": re.compile(
        r"(?i)\bgit\s+(?:-C\s+\S+\s+)?(?:log|show|blame|shortlog|reflog)\b|^commit [0-9a-f]{40}|^Author:\s", re.M
    ),
    "security finding": re.compile(
        r"(?i)\b(?:ssrf|xss|csrf|rce|sqli|sql\s+injection|idor|lfi|cve-\d{4}|vulnerab\w*|zranitel\w*"
        r"|exploit\w*|pentest\w*|penetration\s+test\w*|security\s+(?:finding|audit|review|issue)s?"
        r"|bezpečnostní\s+(?:díra|díry|chyb\w*|audit\w*|problém\w*))\b"
    ),
    "machine inventory": re.compile(
        r"(?i)\b(?:tailscale\s+(?:status|ip)|ip\s+(?:addr|link|route)|ifconfig|nmcli|printenv|hostnamectl"
        r"|arp\s+-a|ss\s+-\w*[tlnp]|netstat|lsusb|lspci|dmidecode)\b|/etc/hosts\b|\.ssh/(?!known_hosts\b)"
    ),
    "price": CURRENCY,
    # what someone said into the dictation tool, as its log records it
    "dictation": re.compile(r"(?i)\bTranscribed:|auto-detected language"),
}
# Reading these is reading a digest of everything: memory files are distilled
# private context, ~/.claude.json holds the account and the MCP inventory, and
# another session's transcript or tool output is any of the above.
PRIVATE_READ = re.compile(r"(?i)\.claude/projects/|/memory/[^\s\"']*\.md|MEMORY\.md|\.claude\.json|mcpServers")


def disqualify(fields: dict) -> str:
    """Why this row must not be kept, or '' if it may. Runs on the raw call,
    before any pseudonym is minted: once invoices.example.cz has become host-1a2b3c4d
    the topic filter can no longer see the word it is looking for."""
    for field_name in ("goal", "said", "input", "result"):
        value = str(fields.get(field_name, ""))
        found = SENSITIVE.search(value)
        if found:
            return found.group(1).lower()
        for reason, pattern in DROP_ROWS.items():
            if pattern.search(value):
                return reason
        if field_name in ("input", "result") and PRIVATE_READ.search(value):
            return "private context"
        # three phone numbers or addresses in one output is a contact list
        people = sum(len(p.findall(value)) for p in (SECRETS[7], SECRETS[-2], SECRETS[-1]))
        if people >= 3:
            return "contact list"
    return ""




# Well-known infrastructure. Their names identify a stack, not a customer, and
# bucketing them would cost the model most of what it knows about the work.
VENDORS = {
    "supabase", "vercel", "netlify", "cloudflare", "openrouter", "anthropic", "openai",
    "digitalocean", "hetzner", "contabo", "coolify", "railway", "render", "sentry",
    "posthog", "plausible", "resend", "twilio", "mailgun", "google", "microsoft",
    "apple", "amazon", "cloudinary", "gravatar", "unsplash", "wikipedia", "wordpress",
    "shopify", "webflow", "squarespace", "tailscale", "ngrok", "sslip", "nip",
    "claude", "docker", "github", "gitlab", "npmjs", "pypi", "kaggle", "huggingface",
    "youtube", "twitter", "facebook", "instagram", "linkedin", "reddit", "medium",
}


def host_labels(text: str) -> set[str]:
    """The brand behind a domain. A client called brewhaus is never a directory
    on this machine, so no environment scan can name it - but it owns
    brewhaus.cz, and that name is in the text. Read the label out of the domain
    and the bare word becomes redactable too."""
    found = set()
    for match in BARE_HOST.finditer(text):
        host = match.group(1).lower()
        bits = host.split(".")
        # wiki.archlinux.org is not in the allowlist but archlinux.org is, and
        # the brand behind both is the same one we agreed not to touch
        if host in PUBLIC_HOSTS or ".".join(bits[-2:]) in PUBLIC_HOSTS:
            continue
        parts = bits[:-1]
        if not parts:
            continue
        # the registrable label is the brand; the ones in front of it are
        # www, api, accounts, and taking those would bucket ordinary words.
        # A long label is kept as well: that is the shape of a generated
        # project reference, which identifies a deployment exactly.
        candidates = {parts[-1]} | {x for x in parts[:-1] if len(x) >= 12}
        found |= {c for c in candidates if len(c) > 3 and c not in GENERIC_NAMES and c not in VENDORS}
    return found


def env_denylist(extra: Path | None = None) -> set[str]:
    """The names only this machine knows: its user, its clients, its hosts.

    `extra` adds names gathered elsewhere. A name known to one machine still
    appears in another machine's transcripts, so the union of every
    contributor's list is what each harvest has to apply.

    No regex guesses an unknown client's name, but the filesystem and the ssh
    config already hold the whole list. Built where the harvest runs, so each
    contributor's own names are covered by their own run.
    """
    curated: set[str] = set()
    if extra and extra.exists():
        curated = {n.lower() for n in extra.read_text().split()}
    # A three-letter name from the curated file is kept: short first names are
    # common in prose. From a directory listing it would be noise.
    curated = {n for n in curated if len(n) >= 3 and n not in GENERIC_NAMES}
    if os.environ.get("JEVLAB_NO_DENYLIST"):  # skips the scan of this machine, not the given file
        return curated
    names: set[str] = {Path.home().name}
    for base in (Path.home() / "Projects", Path.home() / "projects", Path("/var/www"), Path("/srv")):
        try:
            names |= {p.name for p in base.iterdir() if not p.name.startswith(".")}
        except OSError:
            pass
    try:
        config = (Path.home() / ".ssh" / "config").read_text(errors="replace")
        names |= set(re.findall(r"(?im)^\s*Host(?:Name)?\s+(\S+)", config))
    except OSError:
        pass
    # The account names. These identify the person exactly and are never a
    # directory, so no filesystem scan finds them: a git remote or a commit
    # trailer carries them into the transcripts instead.
    for path, pattern in (
        (Path.home() / ".gitconfig", r"(?im)^\s*(?:name|email)\s*=\s*(\S+)"),
        (Path.home() / ".config/git/config", r"(?im)^\s*(?:name|email)\s*=\s*(\S+)"),
        (Path.home() / ".config/gh/hosts.yml", r"(?im)^\s*user:\s*(\S+)"),
    ):
        try:
            for found in re.findall(pattern, path.read_text(errors="replace")):
                names.add(found)
                if "@" in found:  # the local part is what appears in prose
                    names.add(found.split("@")[0])
                # a handle is often written without its suffix: jane-doe -> jane
                names |= {part for part in re.split(r"[-_.]", found) if len(part) > 3}
        except OSError:
            pass
    prefixes = {n.rsplit(".", 1)[0] for n in names if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", n)}
    found = {n.lower() for n in names | prefixes if len(n) > 3 and n.lower() not in GENERIC_NAMES}
    return found | curated


def print_denylist() -> int:
    """Print this machine's names so they can be merged into a shared list."""
    print("\n".join(sorted(env_denylist())))
    return 0


_DENY_RE: tuple[re.Pattern, re.Pattern] | None = None


DENYLIST_FILE: Path | None = None
EXTRA_NAMES: set[str] = set()

# Czech declines a name ("u brewhausu", "Milovi", "northwindový") and English
# adds a plural or a possessive. The ending is matched and replaced with the
# name, so no suffix is left standing next to the bucket to say which one it was.
CZ_SUFFIX = (
    r"(?:ovi|ov[aáéěouýy]|ových|ovými?|ovou|ovsk\w*|ách|ami|ech|ích|ého|ému|ým|ho|mu|em|ům|ou|es"
    r"|[aáeéěiíouůyýs])?"
)
# A short name takes only the endings of a Czech first name. "mat" + "e" is
# "mate", and a bucket that also stands for "mate" is a crib of its own.
SHORT_NAME, SHORT_SUFFIX = 4, r"(?:ovi|ov[aou]|ův|em|ho|mu|ou|a|u)?"
# Long enough that no common word contains it, so it is safe to find inside an
# all-lowercase compound (checknorthwindhome). A short name is found only as a
# word: "mila" is inside "similar", "ella" inside "umbrella".
SUBSTRING_MIN = 7
_LETTER = r"[^\W\d_]"


def _deny_pattern() -> tuple[re.Pattern, re.Pattern]:
    """Two matchers over the same names.

    `word` needs a boundary on both sides: not a letter, or a camelCase seam
    (checkInvoicelyConnection). A plain substring match is invertible, because
    the residue around the bucket rebuilds the word - "siname-5f3a9c1er" is similar, so the bucket is "mila".
    Over-matching is not the safe direction when the neighbours are a crib.

    `inner` finds the long names anywhere, for the compounds with no seam.
    Either way, when a name sits inside a larger token the whole token is
    replaced, so nothing of the word around it survives to be read.
    """
    global _DENY_RE
    if _DENY_RE is None:
        names = sorted(env_denylist(DENYLIST_FILE) | EXTRA_NAMES, key=len, reverse=True)
        alt = "|".join(re.escape(n) for n in names if len(n) > SHORT_NAME)
        short_alt = "|".join(re.escape(n) for n in names if len(n) <= SHORT_NAME)
        long_alt = "|".join(re.escape(n) for n in names if len(n) >= SUBSTRING_MIN)
        never = re.compile(r"(?!)")
        seam = r"(?<=[a-z])(?=[A-Z])"
        forms = [f"(?P<name>(?i:{alt}))(?i:{CZ_SUFFIX})" if alt else "",
                 f"(?P<short>(?i:{short_alt}))(?i:{SHORT_SUFFIX})" if short_alt else ""]
        forms = "|".join(f for f in forms if f)
        word = re.compile(rf"(?:(?<!{_LETTER})|{seam})(?:{forms})(?:(?!{_LETTER})|{seam})") if forms else never
        inner = re.compile(rf"(?P<name>(?i:{long_alt}))") if long_alt else never
        _DENY_RE = (word, inner)
    return _DENY_RE


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _token_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and _is_word_char(text[start - 1]):
        start -= 1
    while end < len(text) and _is_word_char(text[end]):
        end += 1
    return start, end


def replace_names(text: str) -> str:
    word, inner = _deny_pattern()
    spans: list[tuple[int, int, str]] = []
    for m in word.finditer(text):
        start, end = m.span()
        # a hard edge on both sides: the name is a word of its own and only it
        # (with its ending) goes. Otherwise it is part of a compound, and the
        # compound goes with it
        spans.append((*_token_span(text, start, end), m.groupdict().get("name") or m.groupdict().get("short")))
    for m in inner.finditer(text):
        spans.append((*_token_span(text, *m.span()), m.group("name")))
    if not spans:
        return text
    spans.sort()
    merged = [list(spans[0])]
    for start, end, name in spans[1:]:
        if start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, name])
    out, pos = [], 0
    for start, end, name in merged:
        out.append(text[pos:start])
        out.append(_bucket(name, "name-"))
        pos = end
    out.append(text[pos:])
    return "".join(out)


SALT_FILE = Path(__file__).resolve().parent / "salt.txt"
_SALT: bytes | None = None


def _salt() -> bytes:
    """Without this the pseudonyms are decoration. An unsalted sha256 of a name
    drawn from a guessable space (client names, hostnames, project dirs) is
    invertible by hashing a wordlist in seconds. The salt is generated once,
    never committed, and never published - losing it only means old pair files
    stop matching new ones, which a re-harvest fixes."""
    global _SALT
    if _SALT is None:
        if not SALT_FILE.exists():
            SALT_FILE.write_text(secrets.token_hex(32) + "\n")
            SALT_FILE.chmod(0o600)
        _SALT = SALT_FILE.read_text().strip().encode()
    return _SALT


def _bucket(name: str, prefix: str) -> str:
    """A stable stand-in: the same name always maps to the same token, so
    'these two calls touched the same thing' survives while the name does not.
    A constant like [REDACTED] would merge every project into one token and
    take the proxy label's only signal with it."""
    return prefix + hashlib.blake2s(name.lower().encode(), key=_salt()[:32]).hexdigest()[:8]


_PUBLIC_RE = re.compile(
    r"(?i)(?<![\w.-])(" + "|".join(re.escape(h) for h in sorted(PUBLIC_HOSTS, key=len, reverse=True)) + r")(?![\w-])"
)


def _host(name: str) -> str:
    return name if name.lower() in PUBLIC_HOSTS else _bucket(name, "host-")


# This tool prints its own buckets, the terminal output lands in a session
# transcript, and the next harvest reads it back as content. Such a token was
# minted under some earlier salt - possibly none at all - so it is not ours to
# vouch for. It goes before anything this pass creates.
STALE_BUCKET = re.compile(r"(?<![\w-])(?:name|host|proj|sess)-[0-9a-f]{8}(?![\w-])")


def scrub(text: str) -> str:
    text = STALE_BUCKET.sub("[BUCKET]", text)
    text = HOME_PATH.sub(r"\1USER", text)
    # before the names: an address that is also a denylist entry (an IP
    # written into ~/.ssh/config) must become [IP] and not a bucket, because a
    # bucket is one guess away from the address while [IP] is not. The cost is
    # that a hostname embedding a dotted address, like 1.2.3.4.sslip.io, is split
    # before either host pattern sees it.
    text = IPV4.sub("[IP]", text)
    text = MAC.sub("[MAC]", text)
    text = IPV6.sub(_ipv6, text)
    text = HEX_ID.sub("[HEX]", text)
    text = RANDOM_ID.sub("[ID]", text)
    text = PROVIDER_HOST.sub(lambda m: _bucket(m.group(0), "host-"), text)
    # A public host is put beyond reach of the name pass first. Without this a
    # denylisted word inside one defeats the allowlist: an ssh config can have a Host
    # entry literally called github.com, which puts github.com in the denylist and
    # would turn every github URL into a bucket.
    shielded: list[str] = []

    def _shield(match: re.Match) -> str:
        shielded.append(match.group(0))
        return f"\x00{len(shielded) - 1}\x00"

    text = _PUBLIC_RE.sub(_shield, text)
    # the environment's own names, wherever they appear: a path, a git remote,
    # a sentence of Czech prose. This is the class no pattern can find.
    text = replace_names(text)
    text = PROJECT_DIR.sub(lambda m: m.group(1) + _bucket(m.group(2), "proj-"), text)
    text = URL_HOST.sub(lambda m: m.group(1) + _host(m.group(2)), text)
    text = BARE_HOST.sub(lambda m: _host(m.group(1)), text)
    return re.sub(r"\x00(\d+)\x00", lambda m: shielded[int(m.group(1))], text)


def redact(text: str) -> str:
    for pattern in SECRETS:
        text = pattern.sub("[REDACTED]", text)
    return scrub(text)


@dataclass
class Call:
    id: str
    tool: str
    input: dict
    result: str = ""
    is_error: bool = False
    index: int = 0  # position among the session's calls, oldest first
    text_index: int = 0  # assistant texts written before this call, for "what came after"
    ts: str = ""
    narration: str = ""
    answer: dict = field(default_factory=dict)

    @property
    def target(self) -> str:
        """The call's primary argument, for spotting a later repeat of it."""
        for key in ("file_path", "path", "pattern", "command", "url", "query", "notebook_path"):
            value = self.input.get(key)
            if isinstance(value, str):
                return value.strip()
        return json.dumps(self.input, sort_keys=True)[:200]

    def preview(self) -> str:
        return preview(self.result)


def blocks(record: dict) -> list:
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def text_of(block: dict) -> str:
    content = block.get("content", block.get("text", ""))
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else json.dumps(content)[:4000]


def load_session(path: Path) -> tuple[str, list[Call], list[str]]:
    """Returns (goal, calls oldest-first, assistant texts in order)."""
    calls: dict[str, Call] = {}
    order: list[str] = []
    texts: list[str] = []
    goal = ""
    pending_narration = ""
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for block in blocks(record):
            kind = block.get("type")
            if kind == "text":
                body = text_of(block)
                if record.get("type") == "assistant":
                    texts.append(body)
                    pending_narration = body
                elif not goal and body.strip() and not HOST_TEXT.search(body):
                    # cut after redaction, not here: a name split by the cut is
                    # a fragment no pattern recognises any more
                    goal = " ".join(body.split())[:2000]
            elif kind == "tool_use":
                call = Call(
                    id=block.get("id", f"t{len(order)}"),
                    tool=block.get("name", "?"),
                    input=block.get("input") or {},
                    ts=record.get("timestamp", ""),
                    narration=pending_narration,
                    index=len(order),
                    text_index=len(texts),
                )
                calls[call.id] = call
                order.append(call.id)
            elif kind == "tool_result":
                call = calls.get(block.get("tool_use_id", ""))
                if call:
                    call.result = text_of(block)
                    call.is_error = bool(block.get("is_error"))
    return goal or "(no task text found)", [calls[i] for i in order if calls[i].result], texts


def tool_name(tool: str) -> str:
    """The tool as the pair file may show it. An MCP tool's name is the
    server's name, and the list of servers is a private inventory of accounts
    (mcp__some_game_studio__*, mcp__some_mailbox__*); those rows are dropped
    anyway, and anything else unknown is not spelled out either."""
    return tool if tool in BUILTIN_TOOLS else "other"


BUILTIN_TOOLS = REPRODUCIBLE | {
    "Bash", "BashOutput", "Edit", "MultiEdit", "Write", "NotebookEdit", "WebFetch", "WebSearch",
    "Task", "Agent", "Skill", "TodoRead", "KillShell", "KillBash", "ExitPlanMode", "EnterPlanMode",
    "AskUserQuestion", "ToolSearch", "Monitor", "SlashCommand",
}


def questions(call: Call, goal: str = "") -> dict:
    # The goal is in the state already. Interpolating it here repeated it in
    # every row and gave it a second
    # field to leak through. `goal` stays in the signature for the callers.
    tool = tool_name(call.tool)
    return {
        "keep_result": {
            "type": "noul",
            "instructions": (
                f"The agent is still working on the goal in the state. "
                f"It needs the full output of this {tool} call kept verbatim in its context."
            ),
            "criteria": {
                "true": "the contents of this output are still being used for the remaining work",
                "false": "the work this output served is finished, or the output can be fetched again",
            },
        },
        "keep_call": {
            "type": "noul",
            "instructions": (
                f"Knowing that this {tool} call was made, and with which arguments, "
                f"still matters for what the agent does next, even if its output is dropped."
            ),
            "criteria": {
                "true": "the fact this call happened is part of the work so far",
                "false": "the call is noise the agent never needs to know about again",
            },
        },
    }


def preview(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS + 40:
        return text
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{text[:HEAD_CHARS]}\n[... {omitted} chars ...]\n{text[-TAIL_CHARS:]}"


def raw_fields(call: Call, goal: str) -> dict:
    """What `disqualify` reads: the call before redaction, cut a little wider
    than the state so a word near the preview seam is still seen."""
    return {
        "goal": goal,
        "said": call.narration[:600],
        "input": json.dumps(call.input, ensure_ascii=False)[:3000],
        "result": call.result[: HEAD_CHARS + 400] + "\n" + call.result[-(TAIL_CHARS + 400) :],
    }


def state_of(call: Call, goal: str) -> dict:
    return {
        "goal": redact(goal)[:400],
        "tool": tool_name(call.tool),
        "input": preview(redact(json.dumps(call.input, ensure_ascii=False)))[:400],
        "result_chars": len(call.result),
        "failed": call.is_error,
        "result": preview(redact(call.result)),
        # last: Laya keeps the head of the state and cuts the tail, so the least
        # load-bearing field is the one that goes if the budget is ever raised
        "said": redact(" ".join(call.narration.split()))[:200],
    }


def decide(call: Call, keep: float, drop_calls: bool, protect: bool) -> str:
    """keep / truncate / drop, with the non-reproducible guard of issue #25."""
    if call.answer.get("keep_result", 0.0) >= keep:
        return "keep"
    if protect and call.tool not in REPRODUCIBLE:
        return "truncate"
    if not drop_calls or call.answer.get("keep_call", 0.0) >= keep:
        return "truncate"
    return "drop"


def age_baseline(calls: list[Call], preserve: int) -> set[str]:
    """Issue #26: Jev's scores matched 'drop everything but the newest N' almost
    exactly. Any decision model has to beat this rule to be worth its latency."""
    return {c.id for c in calls[: max(0, len(calls) - preserve)]}


# ---- trimming a long tool output before the agent ever sees it (upstream issue #18) ----

TRIM_MIN_CHARS = 4000
CHUNK_LINES, MAX_CHUNKS = 15, 24


def chunks_of(text: str, lines_per: int = CHUNK_LINES, cap: int = MAX_CHUNKS) -> list[str]:
    """Split an output into scoreable pieces, growing the pieces rather than
    letting their number (and so the cost of one trim) run away."""
    lines = text.splitlines()
    step = max(lines_per, -(-len(lines) // cap))
    return ["\n".join(lines[i : i + step]) for i in range(0, len(lines), step)]


def trim_question(goal: str, command: str) -> dict:
    return {
        "keep_chunk": {
            "type": "noul",
            "instructions": (
                f"An agent working on: {goal} just ran `{command[:160]}`. "
                f"This part of the output is worth putting in front of it."
            ),
            "criteria": {
                "true": "it carries the result, an error, a name or a number the agent needs",
                "false": "it is progress noise, repetition, or filler around the real answer",
            },
        }
    }


def head_tail(text: str, budget: int) -> str:
    """The baseline any model-driven trim has to beat: keep the ends, drop the middle."""
    if len(text) <= budget:
        return text
    head = int(budget * 0.7)
    return text[:head] + "\n[...]\n" + text[-(budget - head) :]


def trim_output(agent, text: str, goal: str, command: str, keep: float) -> tuple[str, int]:
    """Returns the trimmed output and how many chunks were kept."""
    pieces = chunks_of(text)
    scores = []
    for piece in pieces:
        state = {"goal": goal, "command": command[:200], "part": piece[:2000]}
        answers = agent.predict(state, trim_question(goal, command))["answers"]
        scores.append(answers["keep_chunk"].get("noul", 0.0))
    out, dropped = [], 0
    for piece, score in zip(pieces, scores):
        if score >= keep:
            out.append(piece)
        else:
            dropped += len(piece)
    if dropped:
        out.append(f"[jevlab trimmed {dropped} chars of this output; re-run to see all of it]")
    return "\n".join(out), sum(1 for s in scores if s >= keep)


def referenced_lines(result: str, later_text: str) -> list[str]:
    """Lines of an output that turn up again in what the assistant wrote next:
    the part it demonstrably used, and so the part a trim must not lose."""
    seen = []
    for line in result.splitlines():
        line = line.strip()
        if 40 < len(line) < 200 and line in later_text and line not in seen:
            seen.append(line)
    return seen


def load_agent():
    os.environ.setdefault("USE_TF", "0")  # model card: TF's abseil deadlocks model construction
    # keep the ~840 MB checkpoint inside the project, so rm -rf really removes everything
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf"))
    import laya

    return laya.load(MODEL)


def cmd_replay(args: argparse.Namespace) -> int:
    goal, calls, _ = load_session(Path(args.session))
    if args.limit:
        calls = calls[: args.limit]
    if not calls:
        print("no completed tool calls in that session")
        return 1
    print(f"session   {args.session}")
    print(f"goal      {goal[:110]}")
    print(f"calls     {len(calls)}  ({sum(len(c.result) for c in calls):,} chars of results)\n")

    agent = load_agent()
    started = time.perf_counter()
    for call in calls:
        result = agent.predict(state_of(call, goal), questions(call, goal))
        call.answer = {k: v.get("noul", 0.0) for k, v in result["answers"].items()}
    elapsed = time.perf_counter() - started

    print(f"{'#':>4} {'tool':<12} {'chars':>8} {'keepRes':>8} {'keepCall':>9}  {'action':<9} target")
    kept = 0
    dropped_by_model = set()
    for call in calls:
        action = decide(call, args.keep, args.drop_calls, not args.no_protect)
        kept += len(call.result) if action == "keep" else (300 if action == "truncate" else 0)
        if action != "keep":
            dropped_by_model.add(call.id)
        print(
            f"{call.index:>4} {call.tool[:12]:<12} {len(call.result):>8,} "
            f"{call.answer.get('keep_result', 0):>8.3f} {call.answer.get('keep_call', 0):>9.3f}  "
            f"{action:<9} {call.target[:44]}"
        )

    total = sum(len(c.result) for c in calls)
    dropped_by_age = age_baseline(calls, args.preserve)
    agree = len(dropped_by_model & dropped_by_age) + len(
        (set(c.id for c in calls) - dropped_by_model) & (set(c.id for c in calls) - dropped_by_age)
    )
    scores = [c.answer.get("keep_result", 0.0) for c in calls]
    print(f"\nlatency        {elapsed / len(calls) * 1000:.0f} ms/call, {elapsed:.1f} s total")
    print(f"result chars   {total:,} -> {kept:,}  ({100 * (1 - kept / max(total, 1)):.0f}% removed)")
    print(f"keep_result    min {min(scores):.3f}  median {sorted(scores)[len(scores)//2]:.3f}  max {max(scores):.3f}")
    print(f"age baseline   agrees on {agree}/{len(calls)} calls ({100*agree/len(calls):.0f}%)")
    if agree / len(calls) > 0.9:
        print("               -> the model is reproducing 'drop the oldest'. Not worth its latency.")
    return 0


def weak_labels(calls: list[Call], texts: list[str], preserve: int = PRESERVE) -> list[tuple[Call, int, str, str]]:
    """Labels harvested from what the session itself did after each call.

    Two kinds, kept apart on purpose. `gold` is behaviour that settles the
    question: the agent edited the file it had read, or fetched the same thing
    again. `proxy` is the cheap dense rule - the call's target comes up again
    later, in a further call or in what the assistant wrote. Train on proxy,
    measure on gold; if they disagree the proxy is what is wrong.
    """
    after = lambda call: "\n".join(texts[call.text_index :])
    out = []
    for i, call in enumerate(calls[: len(calls) - preserve] if len(calls) > preserve else []):
        rest = calls[i + 1 :]
        repeated = any(c.tool == call.tool and c.target == call.target for c in rest)
        edited = call.tool in ("Read", "NotebookRead") and any(
            c.tool in ("Edit", "Write", "NotebookEdit") and c.target == call.target for c in rest
        )
        # a distinctive line of the output turning up in later narration = it was used
        lines = [l.strip() for l in call.result.splitlines() if 40 < len(l.strip()) < 200]
        tail = after(call)
        quoted = any(l in tail for l in lines[:40])
        if edited and not repeated:
            out.append((call, 1, "gold", "edited later without re-reading"))
        elif quoted:
            out.append((call, 1, "gold", "output quoted in later narration"))
        elif repeated:
            out.append((call, 0, "gold", "the agent fetched it again anyway"))
        else:
            name = os.path.basename(call.target.split()[0]) if call.target.split() else ""
            mentioned = len(name) > 3 and (
                name in tail or any(name in json.dumps(c.input) for c in rest)
            )
            out.append(
                (call, int(mentioned), "proxy",
                 "target comes up later" if mentioned else "target never comes up again")
            )
    return out


def session_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*.jsonl") if p.stat().st_size >= 50_000)
    # Not this tool's own sessions. They hold its audit output and the greps
    # written to check it, so every string this tool looks for is guaranteed to
    # be in them - an echo that reappears in the results at every pass.
    own = str(Path(__file__).resolve().parent).replace("/", "-")
    return [f for f in files if own not in str(f)]


def cmd_hosts(args: argparse.Namespace) -> int:
    """Brands read out of the domains in this machine's sessions, with counts.

    A client called brewhaus is never a directory here, so the environment
    scan cannot name it, but it owns brewhaus.cz. Learnt automatically, this
    list also picks up words like `claude`, `docker` or `script` that identify
    nobody. So they are printed, filtered against a dictionary and read by a person, and
    only what survives goes into the shared denylist."""
    root = Path(args.root).expanduser() if args.root else SESSIONS
    seen: collections.Counter = collections.Counter()
    for path in session_files(root):
        for label in host_labels(path.read_text(errors="replace")):
            seen[label] += 1
    known = env_denylist(Path(args.denylist).expanduser() if args.denylist else None)
    for label, n in seen.most_common():
        if label not in known:
            print(f"{n:>5} {label}")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    global DENYLIST_FILE
    if args.denylist:
        DENYLIST_FILE = Path(args.denylist).expanduser()
        print(f"denylist: {len(env_denylist(DENYLIST_FILE))} names")
    root = Path(args.root).expanduser() if args.root else SESSIONS
    files = session_files(root)
    rows, counts = [], {("gold", 0): 0, ("gold", 1): 0, ("proxy", 0): 0, ("proxy", 1): 0}
    dropped: collections.Counter = collections.Counter()
    for path in files:
        try:
            goal, calls, texts = load_session(path)
        except Exception as exc:  # a truncated or in-flight session must not stop the harvest
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            continue
        for call, label, kind, why in weak_labels(calls, texts):
            # An MCP result is someone else's data: a mailbox, a calendar, a
            # contact search. And the tool name alone lists the accounts.
            if call.tool.startswith("mcp__") or tool_name(call.tool) == "other":
                dropped["mcp / unknown tool"] += 1
                continue
            reason = disqualify(raw_fields(call, goal))
            state = state_of(call, goal)
            reason = reason or disqualify(state)
            if reason:
                dropped[reason] += 1
                continue
            counts[(kind, label)] += 1
            rows.append(
                {
                    "state": state,
                    "questions": {k: v for k, v in questions(call).items() if k == "keep_result"},
                    "answers": {"keep_result": {"noul": float(label)}},
                    # no machine name: a source field would carry a host alias
                    # into every row. The file name says whose it is.
                    "label": kind,
                    "label_reason": why,
                    # grouping only (a split must not put one session on both sides)
                    "session": _bucket(path.stem, "sess-"),
                }
            )
    out = Path(args.out)
    out.touch(mode=0o600)
    out.chmod(0o600)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} pairs from {len(files)} sessions -> {out} (own sessions excluded)")
    for kind in ("gold", "proxy"):
        print(f"  {kind:<6} keep {counts[(kind, 1)]:>5}   drop {counts[(kind, 0)]:>5}")
    if dropped:
        total = sum(dropped.values())
        top = ", ".join(f"{t} {n}" for t, n in dropped.most_common(12))
        print(f"  dropped {total} rows: {top}")
    print("Hold the gold rows out of training and score the fine-tune on them.")
    print("Then: audit --denylist, and read a random sample by hand.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    goal = "Fix the failing auth test"
    call = Call(id="t1", tool="Read", input={"file_path": "internal/auth/auth.go"}, index=0)
    call.result = "\n".join(f"{i}: some line of go source here" for i in range(200))
    started = time.perf_counter()
    agent = load_agent()
    load_s = time.perf_counter() - started
    timings = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        agent.predict(state_of(call, goal), questions(call, goal))
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    print(f"model load   {load_s:.1f} s")
    print(f"per call     median {timings[len(timings)//2]:.0f} ms   min {timings[0]:.0f}   max {timings[-1]:.0f}")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"VRAM         {torch.cuda.memory_allocated()/2**20:.0f} MiB allocated, "
                  f"{torch.cuda.memory_reserved()/2**20:.0f} MiB reserved")
    except ImportError:
        pass
    return 0


def cmd_selftest(_: argparse.Namespace) -> int:
    session = Path(sys.argv[0]).parent / ".selftest.jsonl"
    records = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Fix the login test"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Reading the file."},
            {"type": "tool_use", "id": "a", "name": "Read", "input": {"file_path": "auth.go"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "x" * 5000}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "b", "name": "Edit", "input": {"file_path": "auth.go"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "b", "content": "ok"}]}},
    ]
    session.write_text("\n".join(json.dumps(r) for r in records))
    try:
        goal, calls, texts = load_session(session)
        assert goal == "Fix the login test", goal
        assert [c.tool for c in calls] == ["Read", "Edit"], calls
        assert calls[0].target == "auth.go"
        assert len(calls[0].preview()) < 1000, "preview must stay inside the 1024-token window"
        assert "[... 4200 chars ...]" in calls[0].preview()
        # 1024-token window minus the 256-token question head; ~3.5 chars/token on
        # paths and code. Over this and Laya silently drops the end of the state.
        assert len(json.dumps(state_of(calls[0], "g" * 400))) < 2600, "state outgrew the window"

        labels = {c.id: (l, k) for c, l, k, _ in weak_labels(calls, texts, preserve=0)}
        assert labels["a"] == (1, "gold"), labels  # read then edited, never re-read -> was needed
        assert weak_labels(calls, texts) == [], "the newest calls must stay unlabelled"

        assert age_baseline(calls, 1) == {"a"}
        assert age_baseline(calls, 99) == set()

        c = Call(id="x", tool="WebFetch", input={}, answer={"keep_result": 0.1, "keep_call": 0.1})
        assert decide(c, 0.5, True, protect=True) == "truncate", "one-shot result must survive"
        assert decide(c, 0.5, True, protect=False) == "drop"
        c.tool = "Read"
        assert decide(c, 0.5, True, protect=True) == "drop"
        assert decide(c, 0.5, False, protect=True) == "truncate"
        c.answer = {"keep_result": 0.9, "keep_call": 0.0}
        assert decide(c, 0.5, True, protect=True) == "keep"

        # trimming: chunk count stays bounded, and the baseline keeps both ends
        long_output = "\n".join(f"line {i} of output" for i in range(2000))
        pieces = chunks_of(long_output)
        assert len(pieces) <= MAX_CHUNKS, len(pieces)
        assert "\n".join(pieces) == long_output, "chunking must not lose or reorder a line"
        cut = head_tail(long_output, 1000)
        assert len(cut) <= 1010 and cut.startswith("line 0") and cut.endswith("line 1999 of output")
        assert referenced_lines("a" * 50 + "\nshort\n", "prose " + "a" * 50) == ["a" * 50]
        assert referenced_lines("x" * 50, "nothing like it") == []

        # nothing that reaches a training set may carry a credential
        for leak in [
            'export ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnop',
            '{"authorization": "Bearer eyJhbGciOiJIUzI1NiJ9"}',
            "ghp_0123456789abcdefghij",
            "AKIAIOSFODNN7EXAMPLE",
            "password: hunter2hunter2",
            "someone@example.invalid",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQ\n-----END RSA PRIVATE KEY-----",
        ]:
            cleaned = redact(leak)
            assert "[REDACTED]" in cleaned, leak
            for secret in ("sk-ant", "ghp_0123", "AKIAIOSFODNN7", "hunter2", "example.invalid", "MIIEpQ"):
                assert secret not in cleaned, (leak, cleaned)
        assert redact("git status --porcelain") == "git status --porcelain", "must not eat real work"

        # identity and infrastructure
        assert redact("/home/alice/Projects/acme/run.py").startswith("/home/USER/Projects/proj-")
        assert "acme" not in redact("/home/alice/Projects/acme/run.py")
        assert "alice" not in redact("/Users/alice/projects/acme/x.ts")
        assert redact("ssh 100.64.0.7 uptime") == "ssh [IP] uptime"
        assert redact("curl https://github.com/a/b") == "curl https://github.com/a/b", "public host kept"
        # a denylisted name inside a public host must not defeat the allowlist
        _saved, globals()["_DENY_RE"] = _DENY_RE, None
        globals()["EXTRA_NAMES"] = {"github.com", "acme"}
        try:
            assert redact("clone https://github.com/a/b") == "clone https://github.com/a/b"
            assert "acme" not in redact("deploy to acme.cz")
        finally:
            globals()["EXTRA_NAMES"], globals()["_DENY_RE"] = set(), _saved
        assert "acme.com" not in redact("curl https://acme.com/api")
        # the same name must always bucket the same way, or "same target" is lost
        assert _bucket("Acme.com", "host-") == _bucket("acme.com", "host-")
        assert redact("curl https://acme.com/a") != redact("curl https://other.com/a")
        for missed in ["sk_live_abc123", "postgres://u:p@db/x", "PGPASSWORD=swordfish",
                       "curl -u admin:letmein https://x.io", "SG.abcdefghij_klmno"]:
            assert "[REDACTED]" in redact(missed), missed

        # a bare domain in prose never reaches URL_HOST, which needs a scheme
        assert "acme.cz" not in redact("nasadil jsem to na acme.cz a padá to")
        assert "host-" in redact("nasadil jsem to na acme.cz a padá to")
        assert "abcd1234.supabase.co" not in redact("url: abcd1234.supabase.co")
        assert redact("run setup.sh and script.pl") == "run setup.sh and script.pl", "not a host"
        assert redact("see github.com/a/b") == "see github.com/a/b", "public host kept"

        # an address that is also a denylist name must become [IP], not a bucket:
        # a bucket is one guess away from the address it stands for
        assert "[IP]" in redact("Host vps\n  HostName 10.1.2.3")

        # the pseudonyms must not be invertible by hashing a wordlist
        import hashlib as _h
        assert _bucket("acme", "name-") != "name-" + _h.sha256(b"acme").hexdigest()[:8]
        assert SALT_FILE.exists() and len(SALT_FILE.read_text().strip()) >= 32
        # a bucket read back out of an old transcript carries an old salt
        assert redact("audit said name-e0603c49 and proj-dbd551b7") == "audit said [BUCKET] and [BUCKET]"
        assert "[BUCKET]" not in redact("run make-8badf00d now"), "only real bucket prefixes"
        # every field that reaches the file must be scrubbed, not just state
        q = json.dumps(questions(Call(id="q", tool="Read", input={}), "ship acme.cz by friday"))
        assert "acme.cz" not in q, "the goal reaches the question text too"
        assert [p for p, _ in walk_strings({"a": {"b": "x"}, "c": ["y"]})] == ["a.b", "c[0]"]
        # a client brand is learnt from the domain it owns, a vendor is not
        assert host_labels("built brewhaus.cz for them") == {"brewhaus"}
        assert host_labels("accounts.acme.cz") == {"acme"}, "subdomain words are not brands"
        assert host_labels("qwertzuiopasdfghjklm.supabase.co") == {"qwertzuiopasdfghjklm"}
        assert host_labels("x.supabase.co and api.acme.cz") == {"acme"}, "vendor is not a brand"
        assert host_labels("see github.com/a") == set(), "public host is not a brand"
        assert host_labels("wiki.archlinux.org") == set(), "subdomain of a public host"
        # names must survive Czech declension and code compounding
        _saved, globals()["_DENY_RE"] = _DENY_RE, None
        globals()["EXTRA_NAMES"] = {"brewhaus", "northwind", "mila", "ella", "invoicely"}
        try:
            for form in ["u brewhausu", "checkBrewhausConnection", ".northwind_home",
                         "northwindový web", "NORTHWIND_HOME", "mynorthwindapp", "Mila's", "checkInvoicelyConnection"]:
                cleaned = redact(form).lower()
                for name in ("brewhaus", "northwind", "mila", "invoicely"):
                    assert name not in cleaned, (form, cleaned)
            # the whole compound goes: no residue is left to rebuild the word
            assert redact("checkInvoicelyConnection()") == _bucket("invoicely", "name-") + "()"
            assert redact("u brewhausu.") == "u " + _bucket("brewhaus", "name-") + "."
            assert redact("brewhaus-mapper") == _bucket("brewhaus", "name-") + "-mapper"
            # the classic crib: a short name inside a common word
            # must leave the word alone, or the residue names the bucket
            for word in ("similar", "umbrella", "Similarly", "assimilate", "SIMILAR"):
                assert redact(word) == word, (word, redact(word))
            assert redact("Mila said") == _bucket("mila", "name-") + " said"
        finally:
            globals()["EXTRA_NAMES"], globals()["_DENY_RE"] = set(), _saved
        # a three-letter name, from the curated file only, with a first name's endings
        _saved, _saved_file = _DENY_RE, DENYLIST_FILE
        tmp = Path(sys.argv[0]).parent / ".selftest-deny.txt"
        tmp.write_text("jan\n")
        globals()["DENYLIST_FILE"], globals()["_DENY_RE"] = tmp, None
        try:
            for form in ("Jan nechce pauzy", "with Jan's engine", "pro Jana", "Janovi"):
                assert "jan" not in redact(form).lower(), form
            for word in ("January", "janky", "Janus", "Janeiro", "JANUARY"):
                assert redact(word) == word, word
        finally:
            globals()["DENYLIST_FILE"], globals()["_DENY_RE"] = _saved_file, _saved
            tmp.unlink(missing_ok=True)

        # machine and repository identifiers
        assert redact("ether 02:00:5e:12:ab:cd brd") == "ether [MAC] brd"
        assert redact("inet6 fd7a:115c:a1e0::1234/128") == "inet6 [IP]/128"
        assert redact("at 12:30:45 today") == "at 12:30:45 today", "a time is not an address"
        assert redact("x[::-1] and y[::2]") == "x[::-1] and y[::2]", "a slice is not an address"
        assert redact("std::vector") == "std::vector"
        assert redact("abc1234 feat: x") == "[HEX] feat: x"
        assert redact("the deadbeef and defaced") == "the deadbeef and defaced", "no digit, a word"
        assert "vmi1234567" not in redact("root@vmi1234567:~#")
        assert "[REDACTED]" in redact("volejte +420 777 123 456") and "123" not in redact("tel 777 123 456")
        assert redact("chat on acme.chat") != "chat on acme.chat"
        assert redact("./setup.sh && lib.so && req.in") == "./setup.sh && lib.so && req.in"
        assert redact("voice Qx7pL2mR9tWz4kVb8NcA ok") == "voice [ID] ok"
        assert redact("getUser2FactorAuthentication3X") == "getUser2FactorAuthentication3X", "words inside"

        # what the pair file carries besides state
        assert tool_name("mcp__example_server__run") == "other" and tool_name("Bash") == "Bash"
        q = json.dumps(questions(Call(id="q", tool="mcp__mail_server__x", input={}), "ship acme.cz"))
        assert "acme" not in q and "mail_server" not in q

        # redaction runs before truncation: a token split by the 600/200 cut is a
        # token no pattern can recognise any more
        seam = "A" * (HEAD_CHARS - 10) + "ghp_0123456789abcdefghij" + "B" * 4000
        assert "ghp_0123456789" not in preview(redact(seam)), "secret survived the seam"

        # prose leaks meaning, not tokens, so the whole row goes
        assert disqualify({"said": "klientovi jsem poslal fakturu"}) == "klientovi"
        assert disqualify({"result": "cena 990 Kč bez DPH"})
        assert disqualify({"goal": "oprav billing webhook"}) == "billing"
        assert disqualify({"goal": "oprav ten failing test", "result": "ok"}) == ""
        assert disqualify({"result": "91 Jane X 0.97128 leaderboard"}) == "leaderboard"
        assert disqualify({"input": '{"command": "git log --oneline -5"}'}) == "git history"
        assert disqualify({"result": "commit " + "a" * 40 + "\nAuthor: x"}) == "git history"
        assert disqualify({"result": "Registrar: GoDaddy"}) == "whois"
        assert disqualify({"said": "found an SSRF in the upload handler"}) == "security finding"
        assert disqualify({"input": '{"command": "tailscale status"}'}) == "machine inventory"
        assert disqualify({"result": "total $1,200 and $49/mo"}) == "price"
        assert disqualify({"input": '{"command": "awk {print $1}"}'}) == "", "shell $1 is not a price"
        assert disqualify({"result": "a@b.cz\nc@d.cz\ne@f.cz"}) == "contact list"
        assert disqualify({"result": "ls\nBILLING_PLAN.md"}) == "billing"
        assert disqualify({"result": "YC_APPLICATION_NOTES.md"}) == "yc"
        assert disqualify({"result": "checkBilling()"}) == "", "a seam is not checked for topics"
        assert disqualify({"input": '{"command": "cat ~/.claude/projects/x/memory/MEMORY.md"}'}) == "private context"
        assert disqualify({"result": 'INFO Transcribed: "ahoj"'}) == "dictation"
        # the topic filter must see the raw word: after redaction it is a bucket
        raw = raw_fields(Call(id="r", tool="Bash", input={"command": "curl fakturoid.cz/api"}), "g")
        assert disqualify(raw) == "fakturoid"
    finally:
        session.unlink(missing_ok=True)
    print("selftest ok")
    return 0


# Detectors for the audit. Deliberately wider than `redact`: this side over-reports
# so a person reads the samples and decides. Whatever `redact` strips must come
# back with a count of zero here, which is what makes the audit a check and not a
# second opinion.
AUDIT = {
    "api key / token": re.compile(
        r"(?i)\b(sk-[A-Za-z0-9_\-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|xox[abprs]-[A-Za-z0-9\-]{8,}"
        r"|AKIA[0-9A-Z]{12,}|glpat-[\w\-]{12,}|AIza[\w\-]{20,}|hf_[A-Za-z0-9]{12,}|sk_live_\w+)"
    ),
    "bearer / auth header": re.compile(r"(?i)\b(authorization|bearer|x-api-key)\b\s*[:=]?\s*\S{8,}"),
    "secret assignment": re.compile(r"(?i)\b\w*(key|token|secret|password|passwd|credential)\w*\s*[:=]\s*\S{6,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "ssh public key": re.compile(r"\bssh-(rsa|ed25519|dss) [A-Za-z0-9+/]{20,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\."),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"),
    "url with credentials": re.compile(r"\b\w+://[^/\s:@]+:[^/\s@]+@"),
    "db connection string": re.compile(r"(?i)\b(postgres(ql)?|mysql|mongodb(\+srv)?|redis|amqp)://\S+"),
    "ip address": re.compile(r"\b(?!127\.0\.0\.1|0\.0\.0\.0)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ssh private path": re.compile(r"(?i)\B~?/[\w./-]*\.ssh/(?!known_hosts|config\b)[\w.-]+"),
    "home path with user": re.compile(r"(?<![\w.-])/(?:home|Users)/(?!USER\b)[\w.-]+"),
    "unbucketed host": re.compile(
        r"\bhttps?://(?!host-|localhost|127\.)([\w.-]+\.[a-z]{2,})(?<!"
        + r")(?<!".join(re.escape(h) for h in sorted(PUBLIC_HOSTS) if "." in h)
        + r")(?![\w.-])"
    ),
    "credentials file read": re.compile(
        r"(?i)\b(cat|less|more|head|tail|source|\.)\s+\S*(\.env|\.netrc|credentials|"
        r"\.aws/|\.config/gh/|id_rsa|id_ed25519|\.pem|\.key)\b"
    ),
    "crypto address": re.compile(r"\b(bc1[a-z0-9]{25,}|0x[a-fA-F0-9]{40})\b"),
    "mac address": MAC,
    "commit / hex id": HEX_ID,
    "provider hostname": PROVIDER_HOST,
    "phone number": re.compile(SECRETS[-2].pattern + "|" + SECRETS[-1].pattern),
}

def walk_strings(node: object, path: str = "") -> list[tuple[str, str]]:
    """Every string in the row, at any depth. The audit used to read row["state"]
    only, which is how the goal - interpolated raw into the question text - passed
    an audit that reported zero findings."""
    if isinstance(node, str):
        return [(path or "?", node)]
    if isinstance(node, dict):
        return [x for k, v in node.items() for x in walk_strings(v, f"{path}.{k}" if path else str(k))]
    if isinstance(node, list):
        return [x for i, v in enumerate(node) for x in walk_strings(v, f"{path}[{i}]")]
    return []


def cmd_audit(args: argparse.Namespace) -> int:
    """Read every character of a pair file against the detector catalogue.

    Over-reports by design. A regex finds shapes, never meanings, so what it
    cannot see is stated at the end instead of being assumed away.
    """
    path = Path(args.pairs)
    counts = {name: 0 for name in AUDIT}
    samples: dict[str, list[str]] = {name: [] for name in AUDIT}
    denylist = env_denylist(Path(args.denylist).expanduser() if args.denylist else None) if not args.no_denylist else set()
    deny_hits: dict[str, int] = {}
    # the same matcher the harvest removes with: anything it still finds in the
    # output is a name that got through
    global DENYLIST_FILE, _DENY_RE
    DENYLIST_FILE, _DENY_RE = (Path(args.denylist).expanduser() if args.denylist else None), None
    word, inner = _deny_pattern() if denylist else (None, None)
    reasons: collections.Counter = collections.Counter()
    reason_samples: dict[str, str] = {}
    rows = chars = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        rows += 1
        chars += len(line)
        row = json.loads(line)
        for field_name, value in walk_strings(row):
            for name, pattern in AUDIT.items():
                for match in pattern.finditer(value):
                    counts[name] += 1
                    if len(samples[name]) < args.samples:
                        samples[name].append(f"{field_name}: {match.group(0)[:88]}")
            for pattern in (word, inner) if word else ():
                for m in pattern.finditer(value):
                    hit = (m.groupdict().get("name") or m.groupdict().get("short")).lower()
                    deny_hits[hit] = deny_hits.get(hit, 0) + 1
            for m in IPV6.finditer(value):
                if _ipv6(m) != m.group(0):
                    counts["ipv6 address"] = counts.get("ipv6 address", 0) + 1
                    if len(samples.setdefault("ipv6 address", [])) < args.samples:
                        samples["ipv6 address"].append(f"{field_name}: {m.group(0)}")
        why = disqualify(row.get("state", {}))
        if why:
            reasons[why] += 1
            reason_samples.setdefault(why, json.dumps(row.get("state", {}), ensure_ascii=False)[:160])

    print(f"audited {rows:,} rows, {chars:,} characters, {len(AUDIT)} detectors\n")
    worst = 0
    for name in list(AUDIT) + (["ipv6 address"] if "ipv6 address" in counts else []):
        n = counts[name]
        worst = max(worst, n)
        print(f"{'  ' if n == 0 else '!!'} {name:<24} {n:>7,}")
        for sample in samples[name]:
            print(f"        {sample}")
    if reasons:
        worst = max(worst, sum(reasons.values()))
        print(f"\n!! rows the harvest should have dropped: {dict(reasons)}")
        for why, sample in reason_samples.items():
            print(f"        {why}: {sample}")
    if deny_hits:
        top = sorted(deny_hits.items(), key=lambda kv: -kv[1])[:12]
        print(f"\n!! names from your own environment ({len(deny_hits)} of {len(denylist)} appear):")
        print("        " + ", ".join(f"{w}({n})" for w, n in top))
        print("        These are client and machine names. A regex would never guess them;")
        print("        the filesystem and ~/.ssh/config already know them all.")
    print(
        "\nWhat this cannot see: an unreleased product, a private repository's source,\n"
        "a price, a business term, a person named in Czech prose. Those have no shape\n"
        "to match. If the data must be safe against those, it needs reading, not scanning."
    )
    return 0 if worst == 0 and not deny_hits else 2


def cmd_trim(args: argparse.Namespace) -> int:
    """Upstream issue #18, measured: trim a long tool output before the agent sees it.

    The test is not how much it removes - head+tail removes as much as you like.
    It is whether the lines the assistant went on to use survive the trim.
    """
    files = [Path(args.session)] if args.session else sorted(SESSIONS.rglob("*.jsonl"))
    jobs = []
    for path in files:
        if path.stat().st_size < 50_000:
            continue
        goal, calls, texts = load_session(path)
        for call in calls:
            if call.tool == "Bash" and len(call.result) >= TRIM_MIN_CHARS:
                used = referenced_lines(call.result, "\n".join(texts[call.text_index :]))
                if used or not args.used_only:
                    jobs.append((goal, call, used))
        if len(jobs) >= args.limit:
            break
    jobs = jobs[: args.limit]
    if not jobs:
        print("no Bash output over the threshold found")
        return 1
    print(f"{len(jobs)} long Bash outputs, {sum(1 for _, _, u in jobs if u)} with lines used later\n")

    agent = load_agent()
    rows = []
    started = time.perf_counter()
    for goal, call, used in jobs:
        trimmed, kept = trim_output(agent, call.result, goal, call.target, args.keep)
        base = head_tail(call.result, len(trimmed))
        rows.append(
            (
                len(call.result),
                len(trimmed),
                sum(1 for line in used if line in trimmed),
                sum(1 for line in used if line in base),
                len(used),
                kept,
                len(chunks_of(call.result)),
            )
        )
        print(
            f"{len(call.result):>8,} -> {len(trimmed):>7,}  kept {kept:>2}/{rows[-1][6]:<2} chunks  "
            f"used lines kept {rows[-1][2]}/{len(used)} (head+tail {rows[-1][3]}/{len(used)})  "
            f"{call.target[:40]}"
        )

    before = sum(r[0] for r in rows)
    after = sum(r[1] for r in rows)
    used_total = sum(r[4] for r in rows)
    elapsed = time.perf_counter() - started
    print(f"\nchars          {before:,} -> {after:,}  ({100 * (1 - after / before):.0f}% removed)")
    print(f"latency        {elapsed / len(rows) * 1000:.0f} ms per output ({sum(r[6] for r in rows)} chunks total)")
    if used_total:
        laya_recall = sum(r[2] for r in rows) / used_total
        base_recall = sum(r[3] for r in rows) / used_total
        print(f"used lines     laya keeps {laya_recall:.0%}, head+tail at the same size keeps {base_recall:.0%}")
        if laya_recall <= base_recall:
            print("               -> head+tail is free and does the same job. Ship that instead.")
    else:
        print("used lines     none of these outputs were quoted later; rerun with --used-only")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="scan a pair file for anything that must not be published")
    audit.add_argument("pairs", nargs="?", default="pairs-all.jsonl")
    audit.add_argument("--samples", type=int, default=4)
    audit.add_argument("--no-denylist", action="store_true")
    audit.add_argument("--denylist", help="the same shared name list the harvest used")
    audit.set_defaults(func=cmd_audit)

    trim = sub.add_parser("trim", help="trim long Bash output, measured against head+tail")
    trim.add_argument("session", nargs="?")
    trim.add_argument("--limit", type=int, default=25)
    trim.add_argument("--keep", type=float, default=0.5)
    trim.add_argument("--used-only", action="store_true", help="only outputs whose lines were quoted later")
    trim.set_defaults(func=cmd_trim)

    replay = sub.add_parser("replay", help="decide one real session, compare to the age baseline")
    replay.add_argument("session")
    replay.add_argument("--limit", type=int, default=0)
    replay.add_argument("--keep", type=float, default=0.5)
    replay.add_argument("--preserve", type=int, default=6)
    replay.add_argument("--drop-calls", action="store_true", help="allow removing a call, not just its result")
    replay.add_argument("--no-protect", action="store_true", help="let non-reproducible results be dropped too")
    replay.set_defaults(func=cmd_replay)

    harvest = sub.add_parser("harvest", help="weak-labelled keep/drop pairs from your own sessions")
    harvest.add_argument("--out", default="pairs.jsonl")
    harvest.add_argument("--root", help="a session store other than ~/.claude/projects")
    harvest.add_argument("--denylist", help="file of extra names to pseudonymise, one per line")
    harvest.set_defaults(func=cmd_harvest)

    hosts = sub.add_parser("hosts", help="brand names read from the domains in your sessions, to curate")
    hosts.add_argument("--root", help="a session store other than ~/.claude/projects")
    hosts.add_argument("--denylist", help="names already known, left out of the output")
    hosts.set_defaults(func=cmd_hosts)

    bench = sub.add_parser("bench", help="latency and VRAM of one decision")
    bench.add_argument("-n", type=int, default=20)
    bench.set_defaults(func=cmd_bench)

    sub.add_parser("denylist", help="print this machine's identifying names").set_defaults(
        func=lambda _: print_denylist()
    )
    sub.add_parser("selftest", help="parser and baseline maths, no model").set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
