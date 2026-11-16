#!/usr/bin/env python3
"""
Veritas SMS Guard - local web demo
==================================
Run:   python app.py
Open:  http://127.0.0.1:8000   (opens automatically)

Needs veritas_sms_guard.py in the same folder. Uses veritas_model.joblib if it is
there, otherwise trains from SMSSpamCollection automatically (about a minute).

Privacy: the server only listens on your own computer. Messages are never sent
anywhere, and links inside messages are analysed as text only - never opened.
"""
import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import veritas_sms_guard as vsg  # noqa: E402

MODEL_PATH = HERE / "veritas_model.joblib"
DATA_PATH = HERE / "SMSSpamCollection"
MAX_CHARS = 2000
LOCK = threading.Lock()
GUARD = None


def load_guard():
    if MODEL_PATH.exists():
        try:
            return vsg.SmsGuard.load(MODEL_PATH)
        except Exception as exc:  # e.g. scikit-learn version mismatch
            print(f"Could not load the saved model ({exc}). Retraining instead...")
    if not DATA_PATH.exists():
        sys.exit("SMSSpamCollection was not found next to app.py, so the model cannot be trained.")
    print("Training the model (about a minute)...")
    guard, *_ = vsg.train(DATA_PATH, MODEL_PATH, verbose=False)
    return guard


# --------------------------------------------------------------------------- #
# Game messages. Written by hand: modern patterns plus a few ordinary texts.
# --------------------------------------------------------------------------- #
GAME = [
    ("Hey! Are we still on for lunch at 12:30? I'll grab us a table.", "legitimate"),
    ("Your Amazon order #114-2938475 has shipped and will arrive Tuesday. Track it in the Amazon app under Your Orders.", "legitimate"),
    ("USPS: We attempted delivery but nobody was home. Reschedule within 24 hours to avoid return: http://usps-reschedule-pkg.top/track", "scam"),
    ("Chase: Unusual sign-in attempt detected on your account. Confirm your identity now: bit.ly/chase-verify-9x", "scam"),
    ("Hi Mom, I dropped my phone in the lake so this is my new number. Can you send $300 for a bill? I'll pay you back tomorrow.", "scam"),
    ("Your verification code is 739201. Never share this code with anyone. It expires in 10 minutes.", "legitimate"),
    ("FREE ringtones this week only! Text TONE to 85069 to get 5 for $1.50/wk. Reply STOP to cancel.", "spam"),
    ("Dr. Lee's office: reminder of your appointment Thursday at 2:15 PM. Reply C to confirm or call us to reschedule.", "legitimate"),
    ("You've won a $750 Walmart gift card! Claim your prize before it expires: http://walmart-rewards-claim.click/win", "scam"),
    ("Netflix: Your payment was declined. Update your billing info to keep watching: netflix-update-billing.xyz/login", "scam"),
    ("Earn $300 a day from your phone, no experience needed. Message me on Telegram to start today!", "scam"),
    ("Mom, can you pick up Max from practice at 5? My phone is about to die.", "legitimate"),
    ("IRS NOTICE: You owe unpaid taxes. A warrant will be issued unless you call 1-888-555-0142 immediately.", "scam"),
    ("HOT singles in your area want to chat! Reply YES to 62244 to join. 150p/msg. Txt STOP to end.", "spam"),
    ("Your card ending 4417 was used for a $42.17 purchase at Target. If this wasn't you, call the number on the back of your card.", "legitimate"),
]

SCAM_TYPES = {
    "family": ("Family impersonation",
               "Don't send money. Call your family member on the number you already have, not the new one."),
    "delivery": ("Fake delivery notice",
                 "Don't tap the link or pay anything. Check your order in the carrier's real app, or type their website in yourself."),
    "phishing": ("Phishing (account or login theft)",
                 "Don't tap the link. Open the company's app, or type its website yourself, to check your account."),
    "threat": ("Threat / fake authority",
               "Real banks and government agencies don't threaten you by text. Look up their official number yourself and call that."),
    "payment": ("Unusual payment request",
                "Never pay a text with gift cards, crypto or wire transfers. Those payments can't be taken back."),
    "job": ("Job or investment bait",
            "Real jobs don't pay big money for no experience, or move you onto chat apps. Don't reply or share details."),
    "prize": ("Fake prize or giveaway",
              "You can't win a contest you never entered. Don't click, call, or pay a 'fee' to claim it."),
    "premium": ("Premium-rate or subscription trap",
                "Texting a short code or calling that number can sign you up for charges. Delete it, and don't reply, even 'STOP'."),
    "generic": ("Suspected scam", vsg.ADVICE["scam"]),
}

SIMPLE = {
    "scam": "Don't tap anything. This looks like a scam.",
    "spam": "This is junk. It's safe to delete.",
    "legitimate": "Nothing suspicious found. If it asks for money or codes, double-check with the sender another way.",
}

KIND_RANK = {"link": 0, "tactic": 1, "promo": 2}
IP_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def pick_scam_type(label, signals):
    if label == "legitimate":
        return None
    if label == "spam":
        return {"name": "Unwanted marketing", "advice": vsg.ADVICE["spam"]}
    ids = {s.id for s in signals}
    link_flag = any(s.kind == "link" and s.id != "no_https" for s in signals)
    if "family_impersonation" in ids:
        key = "family"
    elif "delivery_lure" in ids:
        key = "delivery"
    elif ids & {"credential_request", "verify_action", "brand_lookalike"} or ("fear_threat" in ids and link_flag):
        key = "phishing"
    elif "fear_threat" in ids:
        key = "threat"
    elif "false_win" in ids:
        key = "prize"
    elif ids & {"payment_request", "money_ask"}:
        key = "payment"
    elif ids & {"easy_money", "offsite_move"}:
        key = "job"
    elif "reward_hype" in ids:
        key = "prize"
    elif ids & {"premium_callback", "shortcode_cta", "curiosity_bait", "cost_notice"}:
        key = "premium"
    else:
        key = "generic"
    name, advice = SCAM_TYPES[key]
    return {"name": name, "advice": advice}


def tactic_span(tac, text):
    """Where in the text a tactic matched (same logic as Tactic.match, but with positions)."""
    for rx in tac._rx:
        for m in rx.finditer(text):
            if tac.skip_if_negated and vsg._NEG.search(text[max(0, m.start() - 45):m.start()]):
                continue
            return m.start(), m.end()
    return None


def link_info(url):
    flags = vsg.analyze_url(url)
    info = {
        "url": url,
        "parts": [],
        "flags": [{"id": s.id, "label": s.label, "why": s.why, "critical": s.critical, "weight": s.weight}
                  for s in flags],
        "brand": None,
        "official": None,
    }
    for f in info["flags"]:
        if f["id"] == "brand_lookalike":
            m = re.search(r"'(.+?)'", f["label"])
            if m:
                info["brand"] = m.group(1)
                official = sorted(vsg.BRANDS.get(info["brand"], []))
                info["official"] = official[0] if official else None
    has_scheme = bool(re.match(r"(?i)^[a-z]+://", url))
    raw = url if has_scheme else "http://" + url
    parts = info["parts"]
    try:
        pr = urlparse(raw)
        host = (pr.hostname or "").lower()
    except ValueError:
        parts.append({"t": url, "role": "rest"})
        return info
    netloc = pr.netloc
    rest = raw[len(pr.scheme) + 3 + len(netloc):]
    if has_scheme:
        parts.append({"t": raw[:len(pr.scheme) + 3], "role": "scheme"})
    hostport = netloc
    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        parts.append({"t": userinfo + "@", "role": "userinfo"})
    hostpart, port = hostport, ""
    if host and hostport.lower().startswith(host):
        hostpart, port = hostport[:len(host)], hostport[len(host):]
    if not host or IP_RE.fullmatch(host):
        parts.append({"t": hostpart, "role": "domain"})
    else:
        registered = vsg._split_host(host)[0]
        sub = hostpart[:len(hostpart) - len(registered)] if hostpart.lower().endswith(registered) else ""
        if sub:
            parts.append({"t": sub, "role": "sub"})
        parts.append({"t": hostpart[len(sub):], "role": "domain"})
    if port:
        parts.append({"t": port, "role": "rest"})
    if rest:
        parts.append({"t": rest, "role": "rest"})
    return info


def make_segments(cleaned, spans):
    cuts = {0, len(cleaned)}
    for s, e, *_ in spans:
        cuts.update((s, e))
    cuts = sorted(cuts)
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        if a == b:
            continue
        cover = [sp for sp in spans if sp[0] <= a and sp[1] >= b]
        if cover:
            cover.sort(key=lambda sp: (KIND_RANK[sp[2]], -sp[3]))
            kind = cover[0][2]
            tip = "; ".join(dict.fromkeys(sp[4] for sp in cover))
        else:
            kind, tip = None, ""
        if segs and segs[-1]["k"] == kind and segs[-1]["tip"] == tip:
            segs[-1]["t"] += cleaned[a:b]
        else:
            segs.append({"t": cleaned[a:b], "k": kind, "tip": tip})
    return segs


def build_result(text):
    with LOCK:
        v = GUARD.classify(text)
    cleaned = vsg.clean_text(text)
    ids = {s.id for s in v.signals}
    spans, links = [], []
    for u in v.urls:
        info = link_info(u)
        links.append(info)
        if info["flags"]:
            i = cleaned.find(u)
            if i >= 0:
                spans.append((i, i + len(u), "link", max(f["weight"] for f in info["flags"]),
                              "Link: " + "; ".join(f["label"] for f in info["flags"])))
    for tac in vsg.TACTICS:
        if tac.id in ids:
            sp = tactic_span(tac, cleaned)
            if sp:
                spans.append((sp[0], sp[1], tac.kind, tac.weight, tac.label))
    stype = pick_scam_type(v.label, v.signals)
    # Keep the meter consistent with the verdict: a "legitimate" verdict never shows 50+.
    risk = min(v.risk_score, 49) if v.label == "legitimate" else v.risk_score
    borderline = v.label == "legitimate" and risk >= 35
    return {
        "label": v.label,
        "borderline": borderline,
        "risk_score": risk,
        "ml_probability": round(v.ml_probability, 4),
        "flagged_by": v.flagged_by,
        "scam_type": stype,
        "advice": stype["advice"] if stype else vsg.ADVICE["legitimate"],
        "simple": ("Probably fine, but double-check before you act on it." if borderline else SIMPLE[v.label]),
        "segments": make_segments(cleaned, spans),
        "signals": [{"id": s.id, "type": s.kind, "finding": s.label, "why": s.why,
                     "evidence": s.evidence, "weight": s.weight} for s in v.signals],
        "links": links,
    }


# --------------------------------------------------------------------------- #
# Web page (single file, no external resources)
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veritas SMS Guard</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%9B%A1%EF%B8%8F%3C/text%3E%3C/svg%3E">
<style>
:root{--bg:#f4f6f9;--card:#fff;--ink:#14171c;--mute:#586271;--line:#e1e6ec;--accent:#2f5bea;--onaccent:#fff;
--ok:#116a3a;--ok-bg:#e3f4ea;--warn:#8f5300;--warn-bg:#fff0d1;--bad:#b42318;--bad-bg:#fde7e4}
@media (prefers-color-scheme:dark){:root{--bg:#0e1217;--card:#161b22;--ink:#e8ecf1;--mute:#9aa6b4;--line:#28303a;
--accent:#7f9fff;--onaccent:#0b0f14;--ok:#62d492;--ok-bg:#10281b;--warn:#f2b45a;--warn-bg:#2b2110;--bad:#ff8f84;--bad-bg:#3a1613}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
body.simple{font-size:20px}
.wrap{max-width:860px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.brand{display:flex;align-items:center;gap:12px}
.logo{font-size:34px}
h1{margin:0;font-size:24px;line-height:1.1}
.tag{margin:2px 0 0;color:var(--mute)}
.toggle{display:flex;align-items:center;gap:8px;color:var(--mute);cursor:pointer}
.toggle input{width:18px;height:18px}
.tabs{display:flex;gap:6px;border-bottom:1px solid var(--line);margin-bottom:16px}
.tabs button{border:0;background:none;color:var(--mute);font:inherit;font-weight:600;padding:10px 14px;cursor:pointer;border-bottom:3px solid transparent}
.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px}
h3{margin:0 0 8px;font-size:17px}
h4{margin:14px 0 4px;font-size:14px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
.lbl{display:block;font-weight:600;margin-bottom:6px}
textarea{width:100%;font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px;resize:vertical}
textarea:focus,button:focus-visible{outline:3px solid var(--accent);outline-offset:1px}
.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}
button.primary{background:var(--accent);color:var(--onaccent);border:0;border-radius:10px;padding:10px 18px;font:inherit;font-weight:700;cursor:pointer}
button.primary:disabled{opacity:.6;cursor:wait}
button.chip{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:5px 12px;font:inherit;font-size:14px;cursor:pointer}
button.chip:hover{border-color:var(--accent)}
.hint,.mute{color:var(--mute)}
.small{font-size:14px}
.banner{display:flex;align-items:center;gap:14px;border-radius:14px;padding:16px;margin-bottom:14px;border:1px solid}
.banner.ok{background:var(--ok-bg);color:var(--ok);border-color:var(--ok)}
.banner.warn{background:var(--warn-bg);color:var(--warn);border-color:var(--warn)}
.banner.bad{background:var(--bad-bg);color:var(--bad);border-color:var(--bad)}
.bicon{font-size:38px}
.btext{flex:1}
.btitle{font-size:24px;font-weight:800;line-height:1.15}
.bsub{font-weight:600}
.risk{min-width:120px;text-align:right}
.rnum{font-weight:800;font-size:20px}
.rbar{height:8px;border-radius:99px;background:rgba(127,127,127,.25);overflow:hidden;margin:4px 0}
.rfill{height:100%;background:currentColor}
.rlab{font-size:12px;opacity:.85}
.simpleline{font-size:1.15em;font-weight:600;margin:0 0 6px}
.bubble{background:var(--bg);border:1px solid var(--line);border-radius:16px;padding:12px 14px;white-space:pre-wrap;word-break:break-word;line-height:1.7}
.bubble.big{font-size:1.1em;margin:10px 0}
mark.hl{background:transparent;color:inherit;border-radius:4px;padding:1px 2px;cursor:help}
mark.hl-link{background:var(--bad-bg);border-bottom:2px solid var(--bad)}
mark.hl-tactic{background:var(--warn-bg);border-bottom:2px dotted var(--warn)}
mark.hl-promo{border-bottom:2px dashed var(--mute)}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--mute)}
.legend mark{font-size:13px}
ul.flags{margin:0;padding-left:20px}
ul.flags li{margin:6px 0}
.why{color:var(--mute);font-size:.92em}
.link{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-top:8px}
.url{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.95em;word-break:break-all}
.u-scheme,.u-sub,.u-rest{color:var(--mute)}
.u-domain{font-weight:800}
.u-userinfo{color:var(--bad);font-weight:700}
.lookalike{color:var(--bad);font-weight:700;margin-top:6px}
.pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.pill{background:var(--warn-bg);color:var(--warn);border-radius:999px;padding:2px 10px;font-size:13px;cursor:help}
.pill.crit{background:var(--bad-bg);color:var(--bad)}
.gprog{color:var(--mute);font-size:14px}
.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:8px}
.choice{font:inherit;font-weight:700;padding:12px 8px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
.choice:hover:not(:disabled){border-color:var(--accent)}
.res{margin-top:12px;padding:12px;border-radius:10px;background:var(--bg);border:1px solid var(--line)}
.err{color:var(--bad);font-weight:600}
footer{color:var(--mute);font-size:13px;margin-top:20px}
body.simple .detail{display:none}
@media (max-width:560px){.risk{min-width:90px}.btitle{font-size:20px}.choices{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand"><span class="logo" aria-hidden="true">🛡️</span>
      <div><h1>Veritas SMS Guard</h1><p class="tag">Think twice before you tap.</p></div></div>
    <label class="toggle"><input type="checkbox" id="simple"> Simple view</label>
  </header>
  <div class="tabs" role="tablist">
    <button role="tab" id="tab-check" aria-selected="true">Check a message</button>
    <button role="tab" id="tab-game" aria-selected="false">Can you beat the AI?</button>
  </div>

  <section id="check">
    <div class="card">
      <label for="msg" class="lbl">Paste a text message</label>
      <textarea id="msg" rows="4" maxlength="2000" placeholder="Paste the message here..."></textarea>
      <div class="row">
        <button id="go" class="primary">Check message</button>
        <span class="hint">Try an example:</span><span id="samples" class="row" style="margin:0"></span>
      </div>
    </div>
    <div id="out" aria-live="polite"></div>
  </section>

  <section id="game" hidden>
    <div class="card" id="gcard"></div>
  </section>

  <footer>Runs entirely on this computer. Your messages are never uploaded and links are never opened.
    Verdicts are a warning to think twice, not a guarantee.</footer>
</div>

<script>
const $ = (s, el = document) => el.querySelector(s);
function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c);
  return e;
}
const LABELS = {
  legitimate: {icon: '✅', title: 'Looks legitimate', cls: 'ok'},
  spam: {icon: '⚠️', title: 'Spam', cls: 'warn'},
  scam: {icon: '🚨', title: 'Likely scam', cls: 'bad'}
};
const SAMPLES = [
  ['Fake delivery', 'USPS: Your package could not be delivered due to an incomplete address. Update your details within 24 hours: http://usps-redelivery-help.xyz/login'],
  ['Bank alert', 'Chase Alert: Your account has been locked due to suspicious activity. Verify now: bit.ly/3xK9aPq'],
  ['Prize', "Congratulations! You've been selected for a $1000 gift card. Claim at http://185.220.101.4/win"],
  ['Real code', 'Your verification code is 482913. Don\'t share this code with anyone. It expires in 10 minutes.']
];

async function classify(text) {
  const res = await fetch('/api/classify', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})});
  if (!res.ok) throw new Error('bad response');
  return res.json();
}

function evidenceBubble(v, extra) {
  const box = h('div', {class: 'bubble' + (extra || '')});
  for (const s of v.segments) {
    if (!s.k) box.append(s.t);
    else box.append(h('mark', {class: 'hl hl-' + s.k, title: s.tip}, s.t));
  }
  return box;
}

function renderEvidence(v) {
  const box = evidenceBubble(v);
  const legend = h('div', {class: 'legend detail'},
    h('span', {}, h('mark', {class: 'hl hl-link'}, 'link red flag')),
    h('span', {}, h('mark', {class: 'hl hl-tactic'}, 'pressure or trick')),
    h('span', {}, h('mark', {class: 'hl hl-promo'}, 'marketing wording')),
    h('span', {text: 'Hover a highlight to see why.'}));
  return h('div', {class: 'card'}, h('h3', {text: 'The message, with evidence'}), box, legend);
}

function renderFlags(v) {
  const groups = [['link', 'Link red flags'], ['tactic', 'Social-engineering tactics'], ['promo', 'Marketing markers']];
  const wrap = h('div', {class: 'card detail'}, h('h3', {text: 'Why it was flagged'}));
  let any = false;
  for (const [k, title] of groups) {
    const items = v.signals.filter(s => s.type === k);
    if (!items.length) continue;
    any = true;
    wrap.append(h('h4', {text: title}),
      h('ul', {class: 'flags'}, items.map(s => h('li', {}, h('strong', {text: s.finding}),
        h('div', {class: 'why', text: s.why})))));
  }
  if (!any) wrap.append(h('p', {class: 'mute', text: 'No red flags found.'}));
  return wrap;
}

function renderLinks(v) {
  if (!v.links.length) return null;
  const wrap = h('div', {class: 'card detail'}, h('h3', {text: 'Link breakdown'}),
    h('p', {class: 'mute small', text: 'The bold part is the real website. Everything before it can be faked. The link is never opened.'}));
  for (const L of v.links) {
    const card = h('div', {class: 'link'},
      h('div', {class: 'url'}, L.parts.map(p => h('span', {class: 'u-' + p.role, text: p.t}))));
    if (L.brand) card.append(h('div', {class: 'lookalike',
      text: 'Pretends to be ' + L.brand + (L.official ? ' (the real site is ' + L.official + ')' : '')}));
    card.append(L.flags.length
      ? h('div', {class: 'pills'}, L.flags.map(f => h('span', {class: 'pill' + (f.critical ? ' crit' : ''), title: f.why, text: f.label})))
      : h('div', {class: 'mute small', text: 'No link red flags found.'}));
    wrap.append(card);
  }
  return wrap;
}

function renderVerdict(v) {
  const L = v.borderline ? {icon: '🤔', title: 'Probably fine', cls: 'warn'} : LABELS[v.label];
  const frag = document.createDocumentFragment();
  frag.append(h('div', {class: 'banner ' + L.cls},
    h('div', {class: 'bicon', 'aria-hidden': 'true', text: L.icon}),
    h('div', {class: 'btext'}, h('div', {class: 'btitle', text: L.title}),
      v.scam_type ? h('div', {class: 'bsub', text: v.scam_type.name}) : null),
    h('div', {class: 'risk'}, h('div', {class: 'rnum', text: v.risk_score + '/100'}),
      h('div', {class: 'rbar'}, h('div', {class: 'rfill', style: 'width:' + Number(v.risk_score) + '%'})),
      h('div', {class: 'rlab', text: 'risk score'}))));
  frag.append(h('div', {class: 'card'}, h('p', {class: 'simpleline', text: v.simple}),
    h('div', {class: 'detail'}, h('strong', {text: 'What to do: '}), v.advice)));
  frag.append(renderEvidence(v), renderFlags(v));
  const links = renderLinks(v);
  if (links) frag.append(links);
  frag.append(h('p', {class: 'mute small detail',
    text: 'Language model: ' + Math.round(v.ml_probability * 100) + '% chance unwanted. Flagged by: ' + v.flagged_by + '.'}));
  return frag;
}

async function check() {
  const text = $('#msg').value.trim();
  const out = $('#out');
  out.replaceChildren();
  if (!text) { out.append(h('p', {class: 'mute', text: 'Paste a message first.'})); return; }
  $('#go').disabled = true;
  try { out.replaceChildren(renderVerdict(await classify(text))); }
  catch (e) { out.append(h('p', {class: 'err', text: 'Could not reach the local server. Is app.py still running?'})); }
  finally { $('#go').disabled = false; }
}

/* ---------- game ---------- */
let G = {};
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

async function startGame() {
  try {
    const items = await (await fetch('/api/game')).json();
    G = {items: shuffle(items), i: 0, you: 0, ai: 0, answered: false};
    renderQ();
  } catch (e) { $('#gcard').replaceChildren(h('p', {class: 'err', text: 'Could not load the game.'})); }
}

function renderIntro() {
  $('#gcard').replaceChildren(
    h('h3', {text: 'Can you beat the AI?'}),
    h('p', {text: 'You and the AI each judge the same messages. Spam is unwanted promotion. A scam tries to trick you out of money or information. Ready?'}),
    h('button', {class: 'primary', onclick: startGame}, 'Start'));
}

function renderQ() {
  const c = $('#gcard');
  c.replaceChildren();
  if (G.i >= G.items.length) { renderEnd(); return; }
  G.answered = false;
  const it = G.items[G.i];
  c.append(
    h('div', {class: 'gprog', text: 'Message ' + (G.i + 1) + ' of ' + G.items.length + '   |   You: ' + G.you + '   AI: ' + G.ai}),
    h('div', {class: 'bubble big', text: it.text}),
    h('strong', {text: 'What is it?'}),
    h('div', {class: 'choices'}, [['legitimate', '✅ Legitimate'], ['spam', '⚠️ Spam'], ['scam', '🚨 Scam']]
      .map(([k, t]) => h('button', {class: 'choice', onclick: () => answer(k)}, t))));
}

async function answer(k) {
  if (G.answered) return;
  G.answered = true;
  const c = $('#gcard');
  c.querySelectorAll('.choice').forEach(b => b.disabled = true);
  const it = G.items[G.i];
  let v;
  try { v = await classify(it.text); }
  catch (e) { c.append(h('p', {class: 'err', text: 'Could not reach the local server.'})); return; }
  const youOk = k === it.answer, aiOk = v.label === it.answer;
  if (youOk) G.you++;
  if (aiOk) G.ai++;
  c.querySelector('.bubble').replaceWith(evidenceBubble(v, ' big'));
  c.append(h('div', {class: 'res'},
    h('div', {}, h('strong', {text: youOk ? '✅ You got it. ' : '❌ Not quite. '}), 'Correct answer: ' + it.answer + '.'),
    h('div', {}, h('strong', {text: aiOk ? '✅ The AI got it. ' : '❌ The AI missed this one. '}),
      'It said: ' + v.label + ' (risk ' + v.risk_score + '/100).'),
    v.scam_type && v.label !== 'legitimate' ? h('div', {class: 'mute small', text: 'Type: ' + v.scam_type.name}) : null),
    renderFlagsInline(v),
    h('div', {class: 'row'}, h('button', {class: 'primary', onclick: () => { G.i++; renderQ(); }},
      G.i + 1 >= G.items.length ? 'See results' : 'Next message')));
}

function renderFlagsInline(v) {
  const box = h('div', {class: 'res'}, h('strong', {text: 'What the AI noticed'}));
  const items = v.signals.filter(s => s.type !== 'promo');
  box.append(items.length
    ? h('ul', {class: 'flags'}, items.map(s => h('li', {text: s.finding})))
    : h('div', {class: 'mute small', text: 'No red flags found.'}));
  return box;
}

function renderEnd() {
  const n = G.items.length;
  $('#gcard').replaceChildren(
    h('h3', {text: 'Results'}),
    h('p', {text: 'You scored ' + G.you + ' out of ' + n + '. The AI scored ' + G.ai + ' out of ' + n + '.'}),
    h('p', {class: 'mute', text: G.you > G.ai ? 'You beat the AI. People still catch things software misses.'
      : G.you === G.ai ? 'A tie.' : 'The AI edged you out, but it can still be fooled. Spotting the tricks yourself is the best defense.'}),
    h('button', {class: 'primary', onclick: startGame}, 'Play again'));
}

/* ---------- wiring ---------- */
for (const [name, text] of SAMPLES) {
  $('#samples').append(h('button', {class: 'chip', onclick: () => { $('#msg').value = text; check(); }}, name));
}
$('#go').addEventListener('click', check);
$('#msg').addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') check(); });
$('#simple').addEventListener('change', e => document.body.classList.toggle('simple', e.target.checked));
function showTab(which) {
  $('#check').hidden = which !== 'check';
  $('#game').hidden = which !== 'game';
  $('#tab-check').setAttribute('aria-selected', which === 'check');
  $('#tab-game').setAttribute('aria-selected', which === 'game');
  if (which === 'game' && !G.items) renderIntro();
}
$('#tab-check').addEventListener('click', () => showTab('check'));
$('#tab-game').addEventListener('click', () => showTab('game'));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the terminal quiet

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/game":
            self._json(200, [{"text": t, "answer": a} for t, a in GAME])
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/classify":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20000:
                return self._json(413, {"error": "message too large"})
            data = json.loads(self.rfile.read(n) or b"{}")
            text = str(data.get("text", "")).strip()[:MAX_CHARS]
        except (ValueError, TypeError):
            return self._json(400, {"error": "bad request"})
        if not text:
            return self._json(400, {"error": "empty message"})
        self._json(200, build_result(text))


def main():
    global GUARD
    ap = argparse.ArgumentParser(description="Veritas SMS Guard web demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    GUARD = load_guard()
    server = None
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Could not find a free port. Try: python app.py --port 9000")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"\nVeritas SMS Guard is running at {url}\nPress Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
