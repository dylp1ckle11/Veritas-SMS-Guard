#!/usr/bin/env python3
"""
Veritas SMS Guard
=================
Classifies a text message as  LEGITIMATE | SPAM | SCAM  and explains why.

Architecture (hybrid, so every verdict is explainable):

  1. ML layer     TF-IDF (word + char n-grams) + style features -> Logistic Regression.
                  Trained on SMSSpamCollection (ham vs spam). Gives P(unwanted).
  2. Link layer   Parses every URL and checks for hallmarks of malicious links
                  (shorteners, raw-IP hosts, punycode, '@' tricks, brand look-alikes,
                  risky TLDs, login-style paths, .apk downloads ...).
  3. Tactic layer Detects social-engineering tactics (urgency, fake prize, authority
                  impersonation, fear/threat, credential requests, payment requests,
                  delivery lures, premium-rate callbacks, family impersonation ...).
  4. Decision     unwanted = ML says so  OR  rules independently find strong evidence.
                  Unwanted + deception signals -> SCAM, otherwise -> SPAM.

The dataset only labels ham/spam. The SCAM tier is therefore *derived* by the link and
tactic layers (weak supervision), not learned from ground-truth scam labels.

Usage
-----
  python veritas_sms_guard.py train   --data SMSSpamCollection --out veritas_model.joblib
  python veritas_sms_guard.py predict "message text" [--json]
  python veritas_sms_guard.py demo
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

DEFAULT_MODEL = Path(__file__).with_name("veritas_model.joblib")

# --------------------------------------------------------------------------- #
# Decision thresholds (rule layer). The ML threshold is learned during training.
# --------------------------------------------------------------------------- #
RULE_ALERT = 50     # rule-only evidence strong enough to flag, even if ML says ham
SCAM_SCORE = 35     # link_risk + deception needed to call an unwanted msg a SCAM


# =========================================================================== #
# Text cleaning
# =========================================================================== #
_ARTIFACTS = re.compile(r"<\s*(?:#|decimal|time|url|email)\s*>", re.I)

# Anti-evasion ("hardening"). Scammers dodge filters with invisible characters and
# look-alike letters (a Cyrillic 'a' in "paypal"). We undo the two easiest tricks.
# Set HARDEN_ENABLED = False to switch it off (used for the before/after experiment).
HARDEN_ENABLED = True
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_HOMOGLYPHS = {
    # Cyrillic lower / upper case
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    # Greek
    "ο": "o", "ν": "v", "Α": "A", "Β": "B", "Ε": "E", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Χ": "X",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPHS)


def harden(text: str) -> str:
    """Normalise compatibility characters, drop zero-width characters, map look-alikes to Latin."""
    t = unicodedata.normalize("NFKC", text)
    return t.translate(_ZERO_WIDTH).translate(_HOMOGLYPH_TABLE)


def clean_text(text: str) -> str:
    """Unescape HTML, undo simple obfuscation, and strip the anonymisation placeholders
    (<#>, <DECIMAL> ...) that appear almost only in ham in SMSSpamCollection (a data leak)."""
    t = html.unescape(text)
    if HARDEN_ENABLED:
        t = harden(t)
    t = _ARTIFACTS.sub(" ", t)
    # re-join obfuscated hosts such as "http://wap. xxxclub.com"
    t = re.sub(r"(?i)((?:https?://|www\.)\S*?\.)\s+(?=\w)", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


_TLDS = (r"com|net|org|co\.uk|uk|biz|info|tv|xyz|top|click|link|tk|ml|ga|cf|gq|ru|cn|ly|gl|"
         r"ws|site|online|live|club|shop|icu|buzz|work|loan|vip|app|ac|pw|su|cyou|rest|zip|mov")
URL_RE = re.compile(
    r"""(?ix)
    (?<![@\w.])
    (?:
        (?:https?://|ftp://)[^\s<>"'()]+
      | www\.[^\s<>"'()]+
      | (?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+(?:""" + _TLDS + r""")(?![a-z0-9@\-])(?:[/?#][^\s<>"'()]*)?
    )"""
)


def find_urls(text: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?)'\"") for m in URL_RE.finditer(text)]


def normalize_for_ml(text: str) -> str:
    t = clean_text(text).lower()
    # Links are removed on purpose: in SMSSpamCollection only 9 of ~4.5k ham messages contain a URL
    # (vs 114 spam), so the model would learn "has a link => spam". Links are judged by the link layer.
    t = URL_RE.sub(" ", t)
    t = re.sub(r"[£$€]\s?\d[\d,\.]*", " moneyamt ", t)
    t = re.sub(r"\d{5,}", " longnum ", t)
    return re.sub(r"\s+", " ", t).strip()


# =========================================================================== #
# ML layer
# =========================================================================== #
class StyleFeatures(BaseEstimator, TransformerMixin):
    """Small, bounded [0,1] stylistic features that survive tokenisation."""

    names = ["length", "digit_ratio", "upper_ratio", "exclaim", "currency",
             "long_number", "shortcode_cta", "caps_words"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for raw in X:
            t = clean_text(raw)
            n = max(len(t), 1)
            letters = [c for c in t if c.isalpha()]
            words = t.split()
            rows.append([
                min(len(t) / 200.0, 1.0),
                sum(c.isdigit() for c in t) / n,
                (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
                min(t.count("!") / 3.0, 1.0),
                1.0 if re.search(r"[£$€]\s?\d", t) else 0.0,
                1.0 if re.search(r"\d{8,}", t) else 0.0,
                1.0 if SHORTCODE_RE.search(t) else 0.0,
                min(sum(1 for w in words if len(w) > 2 and w.isupper()) / 5.0, 1.0),
            ])
        return csr_matrix(np.array(rows, dtype=np.float64))


def build_pipeline(C: float = 10.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize_for_ml, lowercase=False,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize_for_ml, lowercase=False,
                                 analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True)),
        ("style", StyleFeatures()),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear")
    return Pipeline([("features", feats), ("clf", clf)])


# =========================================================================== #
# Link layer
# =========================================================================== #
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy",
              "shorturl.at", "tiny.cc", "buff.ly", "rebrand.ly", "bl.ink", "t.ly", "lnkd.in",
              "s.id", "v.gd", "x.co", "soo.gd", "qr.net"}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "link", "tk", "ml", "ga", "cf", "gq", "icu", "buzz",
                   "loan", "work", "vip", "site", "online", "cyou", "rest", "zip", "mov", "pw",
                   "su", "ru", "cn"}
WEAK_TLDS = {"biz", "info"}
TWO_PART_TLDS = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.in", "co.za", "com.br"}

BRANDS: dict[str, set[str]] = {
    "paypal": {"paypal.com"}, "amazon": {"amazon.com", "amazon.co.uk", "amazon.de", "amazon.in",
                                          "amazon.ca"},
    "apple": {"apple.com", "icloud.com"}, "icloud": {"icloud.com"}, "netflix": {"netflix.com"},
    "usps": {"usps.com"}, "fedex": {"fedex.com"}, "dhl": {"dhl.com", "dhl.de"}, "ups": {"ups.com"},
    "chase": {"chase.com"}, "wellsfargo": {"wellsfargo.com"}, "bankofamerica": {"bankofamerica.com"},
    "citibank": {"citi.com", "citibank.com"}, "microsoft": {"microsoft.com", "live.com", "office.com"},
    "google": {"google.com"}, "irs": {"irs.gov"}, "royalmail": {"royalmail.com"},
    "venmo": {"venmo.com"}, "coinbase": {"coinbase.com"}, "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"}, "whatsapp": {"whatsapp.com"}, "vodafone": {"vodafone.co.uk", "vodafone.com"},
    "barclays": {"barclays.co.uk", "barclays.com"}, "hsbc": {"hsbc.com", "hsbc.co.uk"},
    "lloyds": {"lloydsbank.com"}, "santander": {"santander.co.uk", "santander.com"},
    "zelle": {"zelle.com", "zellepay.com"}, "cashapp": {"cash.app"}, "evri": {"evri.com"},
}
_SHORT_BRANDS = {"ups", "dhl", "irs", "usps"}   # must match a whole host token
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "$": "s"})


@dataclass
class Signal:
    id: str
    kind: str            # "link" | "tactic" | "promo"
    weight: int
    label: str
    why: str
    evidence: str = ""
    critical: bool = False


def _split_host(host: str):
    labels = host.split(".")
    base = 3 if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_PART_TLDS else 2
    return ".".join(labels[-base:]), labels[:-base], labels[-1]


def analyze_url(url: str) -> list[Signal]:
    sigs: list[Signal] = []
    has_scheme = bool(re.match(r"(?i)^[a-z]+://", url))
    try:
        pr = urlparse(url if has_scheme else "http://" + url)
        host = (pr.hostname or "").lower()
        port = pr.port
    except ValueError:
        return [Signal("malformed_url", "link", 15, "Malformed link", "The link cannot be parsed "
                       "cleanly, which is common in obfuscated phishing URLs.", url)]
    if not host:
        return sigs

    def add(id_, w, label, why, critical=False):
        sigs.append(Signal(id_, "link", w, label, why, url[:70], critical))

    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        add("ip_host", 35, "Link points to a raw IP address",
            "Legitimate companies use domain names. Raw IPs are typical of throw-away phishing servers.", True)
    if "xn--" in host:
        add("punycode", 35, "Link uses punycode (look-alike characters)",
            "Punycode can disguise a domain with foreign characters that look like Latin letters.", True)
    if pr.username is not None or "@" in url.split("//", 1)[-1].split("/", 1)[0]:
        add("userinfo_trick", 35, "Link contains an '@' trick",
            "Everything before the '@' is ignored by the browser, so the real destination is hidden.", True)
    if host in SHORTENERS:
        add("shortener", 25, "Shortened link",
            "Link shorteners hide the real destination, so you can't judge it before tapping.")
    registered, subs, tld = _split_host(host)
    if tld in SUSPICIOUS_TLDS:
        add("risky_tld", 20, f"Risky domain ending (.{tld})",
            "These cheap or free domain endings are heavily abused for phishing and scams.")
    elif tld in WEAK_TLDS:
        add("weak_tld", 8, f"Low-trust domain ending (.{tld})",
            "Commonly used by bulk-marketing and scam sites.")
    # brand look-alike: brand name used in the host but the registered domain isn't the brand's
    h = host.translate(_LEET)
    tokens = [t for t in re.split(r"[.\-_]", h) if t]
    for brand, official in BRANDS.items():
        hit = (brand in tokens) if brand in _SHORT_BRANDS else any(
            t == brand or t.startswith(brand) or t.endswith(brand) for t in tokens)
        if hit and not any(host == d or host.endswith("." + d) for d in official):
            add("brand_lookalike", 45, f"Link impersonates '{brand}'",
                f"The domain mentions {brand} but is not owned by {brand}. This is the core "
                "trick of phishing links.", True)
            break
    if host.count("-") >= 3 or max((len(l) for l in host.split(".")), default=0) > 28:
        add("stuffed_domain", 10, "Long, hyphen-stuffed domain",
            "Scam domains often chain keywords like 'secure-login-verify-account'.")
    n_sub = len([s for s in subs if s != "www"])
    if n_sub >= 3:
        add("deep_subdomain", 10, "Unusually deep subdomain chain",
            "Many subdomains are used to push the real domain out of view on small screens.")
    path = (pr.path + "?" + (pr.query or "")).lower()
    if re.search(r"(?:login|signin|sign-in|verify|secure|update|account|confirm|wallet|billing|unlock)", path):
        add("credential_path", 10, "Link path suggests a login / verification page",
            "Pages named 'login', 'verify' or 'update' are where stolen credentials are collected.")
    if path.split("?")[0].endswith((".apk", ".exe", ".scr", ".msi", ".dmg")):
        add("file_download", 30, "Link downloads an app or program file",
            "Direct .apk/.exe downloads are a common route for mobile malware.", True)
    if port not in (None, 80, 443):
        add("odd_port", 10, "Link uses a non-standard port", "Unusual for legitimate consumer websites.")
    if has_scheme and url.lower().startswith("http://"):
        add("no_https", 5, "Link is not encrypted (http)", "Legitimate account pages use https.")
    return sigs


# =========================================================================== #
# Social-engineering tactic layer
# =========================================================================== #
SHORTCODE_RE = re.compile(r"(?i)\b(?:txt|text|reply|send|sms)\b[^.\n]{0,40}?\bto\s*(?:no\s*[:.]?\s*)?\d{4,6}\b")
_NEG = re.compile(r"(?i)(?:never|do not|don'?t|dont|won'?t|will not|not)\b[^.!?]{0,40}$")


@dataclass
class Tactic:
    id: str
    kind: str
    weight: int
    label: str
    why: str
    patterns: list[str]
    critical: bool = False
    skip_if_negated: bool = False
    _rx: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._rx = [re.compile(p, re.I | re.S) for p in self.patterns]

    def match(self, text: str) -> Optional[Signal]:
        for rx in self._rx:
            for m in rx.finditer(text):
                if self.skip_if_negated and _NEG.search(text[max(0, m.start() - 45):m.start()]):
                    continue
                return Signal(self.id, self.kind, self.weight, self.label, self.why,
                              m.group(0).strip()[:60], self.critical)
        return None


_CRED_OBJ = (r"(?:password|pass ?code|pin|otp|one[- ]time (?:code|pass ?code)|verification code|security code|"
             r"cvv|ssn|social security|card (?:number|details)|account (?:number|details|no)|bank details|"
             r"log ?in details|credentials|date of birth|dob|mother'?s maiden)")

TACTICS: list[Tactic] = [
    Tactic("false_win", "tactic", 25, "Unexpected prize / win notice",
           "You can't win a draw you never entered. Fake prizes drive advance-fee fraud and premium-rate traps.",
           [r"\byou(?:'ve|\s+have|\s+are|\s*r)?\s+(?:just\s+)?(?:been\s+)?(?:won|selected|chosen|awarded)(?!'?t)\b",
            r"\b(?:u|you)\s+(?:have\s+|ave\s+)?won(?!'?t)\b", r"\bwinner\b", r"\b(?:has|have)\s+won(?!'?t)\b",
            r"\b(?:ur|your)\s+(?:mobile\s+)?(?:no\.?|number)\b.{0,40}\b(?:won|awarded|selected)\b",
            r"\bguaranteed\b.{0,20}(?:cash|prize|award|£|\$)", r"\bawarded\b", r"\b(?:is|are)\s+yours\b",
            r"\b(?:bonus|caller)\s+prize\b", r"\bunredeemed\b"]),
    Tactic("reward_hype", "tactic", 10, "Prize / reward language",
           "Offers of cash, vouchers or free gifts are bait to get you to click, call or pay.",
           [r"\b(?:congratulations|congrats)\b", r"\b(?:prizes?|jackpot|vouchers?|reward|cash prize)\b",
            r"\bfree (?:gift|entry|holiday|phone|camera|msg|message|ringtone|tone|mobile|membership)\b",
            r"[£$€]\s?\d[\d,]*"]),
    Tactic("urgency", "tactic", 10, "Urgency / pressure",
           "Scammers rush you so you act before thinking or verifying.",
           [r"\burgent(?:ly)?\b", r"\bimmediately\b", r"\basap\b", r"\bright away\b", r"\bact now\b",
            r"\bfinal (?:notice|attempt|chance|reminder|warning)\b", r"\blast chance\b",
            r"\b(?:2nd|3rd|second|third) (?:attempt|notice)\b", r"\bvalid\s*(?:for\s*)?\d+\s*(?:hrs?|hours?)\b",
            r"\bwithin\s+\d+\s*(?:hrs?|hours?|days?|mins?|minutes?)\b", r"\bexpir(?:es?|ing)\b",
            r"\bhurry\b", r"\blimited time\b", r"\btoday only\b", r"\bbefore the lines close\b",
            r"\bcall\b(?:\s+\S+){0,4}?\s+(?:now|immediately)\b"]),
    Tactic("authority", "tactic", 20, "Claims to be a company or authority",
           "Impersonating a bank, carrier, tax agency or 'customer service' borrows trust you'd give the real one.",
           [r"\bvalued (?:network )?customer\b", r"\bnetwork (?:customer|operator)\b",
            r"\bcustomer (?:service|care|loyalty)s?\b", r"\baccount statement\b",
            r"\b(?:security|fraud)\s+(?:alert|team|department|dept)\b",
            r"\byour\s+(?:bank|account|card|debit card|credit card|parcel|package|delivery|order|payment|apple id|"
            r"paypal|netflix|amazon|subscription)\b",
            r"^\s*[\[(]?(?:usps|fedex|dhl|ups|royal mail|hmrc|irs|paypal|netflix|amazon|apple|chase|wells fargo|"
            r"bank of america|microsoft|evri|dpd)\b"]),
    Tactic("delivery_lure", "tactic", 20, "Fake delivery / parcel problem",
           "'Missed delivery' texts are the most common smishing lure today.",
           [r"\b(?:delivery|parcel|package|shipment)\b.{0,60}\b(?:waiting|failed|unable|re-?schedule|held|"
            r"undeliverable|could not|couldn'?t|customs|redeliver|incomplete address)\b"]),
    Tactic("fear_threat", "tactic", 20, "Fear / threat",
           "Threats of lockouts, fines or arrest are used to panic you into complying.",
           [r"\b(?:account|card|access)\b.{0,30}\b(?:suspend(?:ed)?|locked|blocked|deactivated|on hold|disabled)\b",
            r"\b(?:suspicious|unusual|unauthori[sz]ed)\s+(?:activity|login|sign-?in|transaction)\b",
            r"\bcompromised\b", r"\blegal action\b", r"\bwill be arrested\b", r"\barrest warrant\b",
            r"\bwarrant for (?:your )?arrest\b", r"\bbailiff\b", r"\bunpaid\s+(?:tax|taxes|toll|fine|balance|bill)",
            r"\bwill be (?:closed|terminated|cancelled|canceled|disconnected)\b"]),
    Tactic("verify_action", "tactic", 25, "Asks you to 'verify' or 'update' something",
           "Real companies rarely ask you to confirm account details through a text link.",
           [r"\b(?:verify|confirm|validate|unlock|reactivate|restore|update)\b.{0,30}\b"
            r"(?:account|identity|information|details|access|card|payment|address|now)\b"],
           skip_if_negated=True),
    Tactic("credential_request", "tactic", 30, "Asks for passwords, PINs or personal data",
           "Genuine banks and carriers never ask for passwords, PINs or one-time codes by text.",
           [r"\b(?:verify|confirm|update|validate|re-?enter|provide|send|share|reply with|give)\b.{0,40}?\b" + _CRED_OBJ + r"\b",
            r"\b(?:your|ur)\s+(?:name and address|address and dob|dob)\b"],
           critical=True, skip_if_negated=True),
    Tactic("payment_request", "tactic", 25, "Asks for unusual payment",
           "Gift cards, wire transfers, crypto and 'fees' are favoured because they can't be reversed.",
           [r"\b(?:gift ?cards?|itunes card|google play card|steam card|western union|moneygram|wire transfer|"
            r"bitcoin|crypto(?:currency)?|btc|cash ?app|zelle|processing fee|registration fee|release fee|"
            r"transfer fee)\b"], critical=True),
    Tactic("money_ask", "tactic", 15, "Asks you to send a specific amount of money",
           "Direct requests for a cash transfer from an unverified sender are a hallmark of impersonation scams.",
           [r"\b(?:send|transfer|lend|loan|wire)\s+(?:me\s+)?(?:about\s+)?[£$€]\s?\d"]),
    Tactic("family_impersonation", "tactic", 30, "'New number' family impersonation",
           "'Hi Mum, I lost my phone' is a well-known scam: criminals pose as a relative to ask for money.",
           [r"\b(?:mum|mom|dad|nan|grandma|grandpa)\b.{0,60}\b(?:new number|lost my phone|broke my phone|"
            r"phone (?:broke|is broken|died)|dropped my phone)\b",
            r"\b(?:new number|lost my phone|broke my phone)\b.{0,60}\b(?:mum|mom|dad)\b"], critical=True),
    Tactic("secrecy", "tactic", 10, "Asks for secrecy",
           "Isolating you from people who could spot the scam is a classic manipulation tactic.",
           [r"\bdon'?t tell (?:anyone|anybody)\b", r"\bkeep (?:this|it) (?:a )?(?:secret|between us|private|confidential)\b",
            r"\btell no ?one\b"]),
    Tactic("curiosity_bait", "tactic", 15, "Curiosity / secret-admirer bait",
           "'Someone fancies you' hooks make you call a premium-rate number to 'find out who'.",
           [r"\bsecret admirer\b", r"\bsomeone (?:u |you )?know\b.{0,40}\b(?:contact|fancies|asked)\b",
            r"\bfancies you\b", r"\bfind out who\b", r"\bdating service\b", r"\breveal who\b"]),
    Tactic("easy_money", "tactic", 30, "Too-good-to-be-true earnings / investment",
           "Guaranteed income, 'no experience' jobs and investment tips are staples of job and crypto scams.",
           [r"\bearn\s+[£$€]?\s?\d[\d,]*\s*(?:/|per|a)?\s*(?:day|daily|hour|hr|week|weekly)\b",
            r"\bno experience (?:needed|required|necessary)\b", r"\binvestment opportunity\b",
            r"\bguaranteed returns?\b", r"\bdouble your money\b"]),
    Tactic("offsite_move", "tactic", 15, "Pushes you onto another app",
           "Moving the chat to WhatsApp/Telegram takes it out of reach of carrier spam filters.",
           [r"\b(?:message|text|contact|add|dm)\s+(?:me\s+)?(?:on|via)\s+(?:whatsapp|telegram|signal)\b",
            r"\b(?:whatsapp|telegram)\b.{0,20}\b(?:to start|to apply|to join)\b"]),
    Tactic("premium_callback", "tactic", 20, "Asks you to call a premium-rate number",
           "UK 09xx / 0871-0873 numbers charge you heavily per minute; calling is how the scam makes money.",
           [r"(?<!\d)(?:\+?44\s?)?0?9[01]\d{8,9}", r"(?<!\d)0?87[0-3]\d{7,8}"]),
    Tactic("shortcode_cta", "tactic", 10, "Asks you to text a short code",
           "Texting a 5-digit short code often enrols you in a recurring paid subscription.",
           [SHORTCODE_RE.pattern]),
    Tactic("cost_notice", "promo", 10, "Per-message / per-minute charges",
           "Fine print about pence-per-message or per-minute rates signals a paid-service trap.",
           [r"\b\d{2,3}p(?:pm|/?(?:msg|min|wk|txt|text|per))?\b", r"\b(?:ppm|p/min|p per min|per min(?:ute)?)\b",
            r"[£$]\s?\d+(?:\.\d+)?\s?(?:/|per|a)\s?(?:wk|week|msg|min|day|month)\b", r"\bsubscri(?:be|bed|ption)\b",
            r"\b(?:charged|billed)\b"]),
    Tactic("optout_language", "promo", 5, "Opt-out / T&C boilerplate",
           "Bulk marketing texts carry STOP / T&C wording.",
           [r"\b(?:txt|text|reply)\s+stop\b", r"\bstop to (?:end|opt out|cancel|unsubscribe)\b", r"\bunsubscribe\b",
            r"\bopt[- ]?out\b", r"\bt&c'?s?\b", r"\btsandcs\b", r"\bterms (?:and|&) conditions\b"]),
    Tactic("marketing_offer", "promo", 5, "Bulk marketing content",
           "Ringtones, handset upgrades, adult chat and competitions are typical bulk-spam themes.",
           [r"\bringtones?\b", r"\bpoly(?:phonic)?\b", r"\bwallpaper\b", r"\blogo\b", r"\bsexy\b", r"\bhorny\b",
            r"\bsingles?\b", r"\bchat line\b", r"\bupgrade\b", r"\bcamcorder\b", r"\bline rental\b",
            r"\bdouble mins\b", r"\bnew (?:video )?(?:phone|mobile|handset)\b", r"\bweekly (?:comp|draw)\b",
            r"\bwkly comp\b", r"\bchance to win\b", r"\bquiz\b", r"\bdogging\b"]),
]


# =========================================================================== #
# Analysis + decision
# =========================================================================== #
@dataclass
class Analysis:
    signals: list[Signal]
    link_risk: int
    deception: int
    promo: int
    urls: list[str]

    @property
    def rule_risk(self) -> int:
        return min(100, self.link_risk + self.deception)

    @property
    def critical(self) -> bool:
        return any(s.critical for s in self.signals)


def analyze(text: str) -> Analysis:
    t = clean_text(text)
    urls = find_urls(t)
    link: dict[str, Signal] = {}
    for u in urls:
        for s in analyze_url(u):
            if s.id not in link or s.weight > link[s.id].weight:
                link[s.id] = s
    signals = list(link.values())
    for tac in TACTICS:
        s = tac.match(t)
        if s:
            signals.append(s)
    return Analysis(
        signals=sorted(signals, key=lambda s: -s.weight),
        link_risk=min(100, sum(s.weight for s in signals if s.kind == "link")),
        deception=sum(s.weight for s in signals if s.kind == "tactic"),
        promo=sum(s.weight for s in signals if s.kind == "promo"),
        urls=urls,
    )


def decide(p_unwanted: float, a: Analysis, ml_threshold: float):
    """Return (label, risk_score 0-100, flagged_by)."""
    by_ml = p_unwanted >= ml_threshold
    by_rules = a.rule_risk >= RULE_ALERT
    risk = 100 * (1 - (1 - p_unwanted) * (1 - a.rule_risk / 100))
    if not (by_ml or by_rules):
        return "legitimate", round(risk), "none"
    scam_score = a.link_risk + a.deception
    label = "scam" if (scam_score >= SCAM_SCORE or a.critical) else "spam"
    flagged_by = "ml+rules" if (by_ml and by_rules) else ("ml" if by_ml else "rules")
    return label, round(max(risk, 50 if by_rules else 50)), flagged_by


ADVICE = {
    "scam": "Do NOT tap links, call numbers or reply (even 'STOP'). If it claims to be a company, "
            "contact them through their official app or website instead. Report and delete.",
    "spam": "Unsolicited promotion. Safe to delete. Texting a short code or replying can enrol you "
            "in paid services.",
    "legitimate": "No strong red flags. Still verify any unexpected request through a channel you trust.",
}


@dataclass
class Verdict:
    label: str
    risk_score: int
    ml_probability: float
    flagged_by: str
    signals: list[Signal]
    urls: list[str]
    advice: str

    def to_dict(self):
        return {
            "label": self.label, "risk_score": self.risk_score,
            "ml_probability_unwanted": round(self.ml_probability, 4), "flagged_by": self.flagged_by,
            "urls": self.urls,
            "signals": [{"id": s.id, "type": s.kind, "weight": s.weight, "finding": s.label,
                         "why_it_matters": s.why, "evidence": s.evidence} for s in self.signals],
            "advice": self.advice,
        }

    def pretty(self) -> str:
        icon = {"scam": "🚨", "spam": "⚠️ ", "legitimate": "✅"}[self.label]
        out = [f"{icon} {self.label.upper()}   (risk {self.risk_score}/100, "
               f"ML P(unwanted)={self.ml_probability:.2f}, flagged by: {self.flagged_by})"]
        links = [s for s in self.signals if s.kind == "link"]
        tactics = [s for s in self.signals if s.kind == "tactic"]
        promo = [s for s in self.signals if s.kind == "promo"]
        for title, group in (("Link red flags", links), ("Social-engineering tactics", tactics),
                             ("Marketing markers", promo)):
            if group:
                out.append(f"  {title}:")
                out += [f"    - {s.label}  [{s.evidence}]" for s in group]
        out.append(f"  Advice: {self.advice}")
        return "\n".join(out)


class SmsGuard:
    def __init__(self, pipeline: Pipeline, ml_threshold: float):
        self.pipeline = pipeline
        self.ml_threshold = ml_threshold

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def classify(self, text: str) -> Verdict:
        p = float(self.predict_proba([text])[0])
        a = analyze(text)
        label, risk, by = decide(p, a, self.ml_threshold)
        return Verdict(label, risk, p, by, a.signals, a.urls, ADVICE[label])

    def save(self, path=DEFAULT_MODEL):
        joblib.dump({"pipeline": self.pipeline, "ml_threshold": self.ml_threshold}, path)

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        d = joblib.load(path)
        return cls(d["pipeline"], d["ml_threshold"])


# =========================================================================== #
# Training / evaluation
# =========================================================================== #
def load_dataset(path):
    rows, seen, conflicts = [], {}, set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        key = normalize_for_ml(txt)
        if key in seen and seen[key] != lab:
            conflicts.add(key)
        seen.setdefault(key, lab)
    out, used = [], set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        lab, txt = line.split("\t", 1)
        key = normalize_for_ml(txt)
        if key in used or key in conflicts:
            continue
        used.add(key)
        out.append((1 if lab == "spam" else 0, txt))
    return out


def pick_threshold(y, p, max_ham_fpr=0.005):
    """Highest-F1 threshold subject to a ceiling on the ham false-positive rate."""
    best_t, best_f1 = 0.5, -1
    ham = max((y == 0).sum(), 1)
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        if fp / ham > max_ham_fpr:
            continue
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def train(data_path, out_path=DEFAULT_MODEL, seed=42, verbose=True):
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline

    rows = load_dataset(data_path)
    y = np.array([r[0] for r in rows]); X = np.array([r[1] for r in rows], dtype=object)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    log = print if verbose else (lambda *a, **k: None)
    log(f"Messages after de-duplication: {len(X)}  (spam={y.sum()}, ham={(y==0).sum()})")
    log(f"Train={len(Xtr)}  Test={len(Xte)}  (stratified, seed={seed})\n")

    # --- model selection by 5-fold CV on TRAIN only ---------------------------------
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = None
    for C in (3.0, 10.0, 30.0):
        oof = cross_val_predict(build_pipeline(C), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        ap = average_precision_score(ytr, oof)
        log(f"  CV  C={C:<5}  PR-AUC={ap:.4f}")
        if best is None or ap > best[0]:
            best = (ap, C, oof)
    _, C, oof = best
    thr = pick_threshold(ytr, oof)
    log(f"\nChosen C={C}; ML threshold={thr:.2f} (max F1 with ham FPR <= 0.5% on CV predictions)\n")

    pipe = build_pipeline(C).fit(Xtr, ytr)
    guard = SmsGuard(pipe, thr)
    p = guard.predict_proba(Xte)

    def report(name, pred, scores=None):
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        res = dict(model=name, precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                   f1=f1_score(yte, pred), ham_fpr=fp / (fp + tn), tp=int(tp), fp=int(fp), fn=int(fn),
                   tn=int(tn))
        if scores is not None:
            res.update(roc_auc=roc_auc_score(yte, scores), pr_auc=average_precision_score(yte, scores))
        log(f"{name:<34} P={res['precision']:.4f} R={res['recall']:.4f} F1={res['f1']:.4f} "
            f"hamFPR={res['ham_fpr']:.4f}  [TP={tp} FP={fp} FN={fn} TN={tn}]"
            + (f"  ROC-AUC={res['roc_auc']:.4f}" if scores is not None else ""))
        return res

    # --- held-out test ------------------------------------------------------------------
    log("=== Held-out TEST set (binary: unwanted vs legitimate) ===")
    results = {}
    nb = make_pipeline(CountVectorizer(preprocessor=normalize_for_ml, lowercase=False), MultinomialNB()).fit(Xtr, ytr)
    results["baseline_nb"] = report("Baseline: Naive Bayes (BoW)", nb.predict(Xte), nb.predict_proba(Xte)[:, 1])
    results["ml_only"] = report("ML layer only", (p >= thr).astype(int), p)

    analyses = [analyze(t) for t in Xte]
    verdicts = [decide(pi, a, thr) for pi, a in zip(p, analyses)]
    hybrid_pred = np.array([v[0] != "legitimate" for v in verdicts], dtype=int)
    results["hybrid"] = report("Hybrid (ML + link/tactic rules)", hybrid_pred)
    rules_only = np.array([a.rule_risk >= RULE_ALERT for a in analyses], dtype=int)
    results["rules_only"] = report("Rules layer only", rules_only)

    labs = np.array([v[0] for v in verdicts])
    tiers = {k: int(((labs == k) & (yte == 1)).sum()) for k in ("spam", "scam")}
    tiers_ham = {k: int(((labs == k) & (yte == 0)).sum()) for k in ("spam", "scam")}
    log(f"\nTier split of true-spam test messages: {tiers}   (flagged ham: {tiers_ham})")
    results["tiers_on_true_spam"] = tiers
    results["flagged_ham_by_tier"] = tiers_ham

    # --- explainability coverage on spam ------------------------------------------------
    with_sig = np.mean([len(a.signals) > 0 for a, t in zip(analyses, yte) if t == 1])
    log(f"Share of true spam with >=1 explainable signal: {with_sig:.1%}")
    results["spam_with_signal"] = float(with_sig)
    results.update(ml_threshold=thr, C=C, n_train=len(Xtr), n_test=len(Xte))

    # --- refit on ALL data for the shipped model -----------------------------------------
    final = SmsGuard(build_pipeline(C).fit(X, y), thr)
    final.save(out_path)
    log(f"\nSaved model -> {out_path}")
    return final, results, (Xte, yte, p, analyses, labs)


# =========================================================================== #
# Demo set: modern smishing patterns the 2000s-era dataset barely contains.
# Hand-written by the author while building the rules: illustrative, NOT a benchmark.
# =========================================================================== #
DEMO = [
    ("USPS: Your package could not be delivered due to an incomplete address. Update your details within 24 hours: http://usps-redelivery-help.xyz/login", "scam"),
    ("Chase Alert: Your account has been locked due to suspicious activity. Verify now: bit.ly/3xK9aPq", "scam"),
    ("Hi Mum, I've lost my phone and this is my new number. Can you send me £250 for a bill? I'll pay you back tomorrow", "scam"),
    ("Your Netflix account is on hold. Update your payment method at netflix-billing-support.top to avoid cancellation", "scam"),
    ("Congratulations! You've been selected for a $1000 gift card. Claim at http://185.220.101.4/win", "scam"),
    ("URGENT: Final notice. IRS has issued an arrest warrant. Call 08712300220 immediately to resolve unpaid taxes.", "scam"),
    ("Earn $500 daily working from home, no experience needed. Message me on WhatsApp to start today!", "scam"),
    ("Get the new season ringtones FREE! Reply TONE to 85069. 150p/week, txt STOP to end.", "spam"),
    ("Your verification code is 482913. Don't share this code with anyone. It expires in 10 minutes.", "legitimate"),
    ("Hi, this is Dr. Patel's office confirming your appointment tomorrow at 3:30 PM. Reply C to confirm.", "legitimate"),
    ("Your Amazon order has shipped and will arrive Tuesday. Track it at https://www.amazon.com/progress-tracker", "legitimate"),
    ("Hey, running 10 min late, can you grab us a table?", "legitimate"),
    ("Chase: A $42.17 purchase at TARGET was made on your debit card ending 4417. If not you, call the number on the back of your card.", "legitimate"),
    ("Congrats on the new job!! Dinner is on me tonight", "legitimate"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Veritas SMS Guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True)
    t.add_argument("--out", default=str(DEFAULT_MODEL)); t.add_argument("--metrics")
    p = sub.add_parser("predict"); p.add_argument("text"); p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--json", action="store_true")
    d = sub.add_parser("demo"); d.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args(argv)

    if a.cmd == "train":
        _, res, _ = train(a.data, a.out)
        if a.metrics:
            Path(a.metrics).write_text(json.dumps(res, indent=2, default=float))
    elif a.cmd == "predict":
        v = SmsGuard.load(a.model).classify(a.text)
        print(json.dumps(v.to_dict(), indent=2, ensure_ascii=False) if a.json else v.pretty())
    else:
        g = SmsGuard.load(a.model); ok = 0
        for text, want in DEMO:
            v = g.classify(text); ok += v.label == want
            print(f"[{'OK ' if v.label == want else 'MISS'}] expected={want:<10} got={v.label}\n  \"{text[:95]}\"")
        print(f"\n{ok}/{len(DEMO)} correct on the illustrative demo set")


if __name__ == "__main__":
    import veritas_sms_guard as _self   # so pickles reference the real module name
    _self.main()
