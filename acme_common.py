"""
acme_common.py -- helpers shared by the Act / Assert / Adjust / Answer scripts.

The idea, in the same terms as microgpt.py:

  * microgpt looks at the text so far and gives every possible next character
    a score (a "logit"), then softmax turns the scores into probabilities.
  * Here we look at ONE TICKET and give every possible ORDER a score, then
    softmax turns the scores into probabilities.

The score is a weighted sum of a few FEATURES: numbers between 0 and 1 that say
"how well does this order fit what the ticket says?" (PO matches, product
matches, the buyer's name appears in the message, and so on).

There is one extra choice on the list: NONE, meaning "no order in our system
matches". Picking NONE is how the model says "I can't link this, ask a human."

Only the standard library is used.
"""
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

NONE = "NONE"

# Features describing how well one order fits one ticket ...
ORDER_FEATURES = ["po_match", "order_id_match", "product_match",
                  "buyer_name_match", "company_match", "email_domain_match",
                  "date_fit"]
# ... and two that only the NONE choice has.
NONE_FEATURES = ["none_bias", "unmatched_ref"]
FEATURES = ORDER_FEATURES + NONE_FEATURES

FEATURE_LABELS = {
    "po_match": "PO number matches",
    "order_id_match": "order number matches",
    "product_match": "product matches",
    "buyer_name_match": "buyer's name appears in the message",
    "company_match": "company name matches",
    "email_domain_match": "sender's email domain matches the account",
    "date_fit": "order date fits 'last month' style wording",
    "none_bias": "default pull toward 'no match'",
    "unmatched_ref": "ticket quotes a PO/order number we can't find",
}

# The policy lines the CS team works with (used by Assert and Answer).
AUTO_THRESHOLD = 0.90    # at or above: one-click suggestion
REVIEW_THRESHOLD = 0.30  # between the two: rep picks from the top 3

# Rough meaning of the time words people use: (youngest, oldest) order age in
# days that "fits". Hand-written guesses, and one place where a smarter
# technique could help later.
TIME_WINDOWS = {
    "last week": (0, 12),
    "last month": (12, 50),
    "a few months ago": (40, 130),
    "earlier this year": (110, 400),
}


# ----------------------------------------------------------------------
# Loading data
# ----------------------------------------------------------------------
def read_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run arrange_data.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_system_data():
    """Everything Acme's systems know, plus the tickets. NOT the answer key."""
    names = ("accounts", "contacts", "products", "orders", "tickets")
    return {name: read_csv(f"{name}.csv") for name in names}


def load_answer_key():
    """Only Assert and Adjust (for training labels) should call this."""
    return {row["ticket_id"]: row for row in read_csv("answer_key.csv")}


def split_ticket_ids(tickets, test_fraction=0.3, seed=7):
    """Hold some tickets back for testing. T-0001 (Jane) is always held back,
    so she is a case the learner never trains on."""
    ids = [t["ticket_id"] for t in tickets if t["ticket_id"] != "T-0001"]
    random.Random(seed).shuffle(ids)
    n_test = round(len(ids) * test_fraction)
    return set(ids[n_test:]), set(ids[:n_test]) | {"T-0001"}


def save_weights(weights, filename):
    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump({k: round(v, 4) for k, v in weights.items()}, f, indent=2)


def load_weights(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Reading a ticket (simple rules; an LLM could do this better later)
# ----------------------------------------------------------------------
def norm(text):
    """Lowercase, strip punctuation, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()  # 1.0 = identical


def best_window_similarity(tokens, target):
    """How closely does `target` appear in the text? Slide a window with the
    same number of words along the text and keep the best score. Typos cost a
    little ('coyte' still scores high) instead of failing outright."""
    n = len(target.split())
    best = 0.0
    for i in range(len(tokens) - n + 1):
        best = max(best, similarity(" ".join(tokens[i:i + n]), target))
    return best


def scaled(sim, floor):
    """Turn a similarity into a 0..1 feature: below `floor` counts as 0."""
    return max(0.0, (sim - floor) / (1.0 - floor))


def date_fit(age_days, window):
    if window is None:
        return 0.0
    lo, hi = window
    if lo <= age_days <= hi:
        return 1.0
    gap = lo - age_days if age_days < lo else age_days - hi
    return max(0.0, 1.0 - gap / 30)  # fades out over about a month


def read_ticket(ticket, accounts, contacts, products):
    """Pull the useful clues out of the free text and form fields."""
    text = ticket["message"]
    low = text.lower()
    # Drop the sender's own name so it isn't mistaken for the buyer's.
    tokens = norm(text.replace(ticket["submitter_name"], " ")).split()

    hints = {
        "po_numbers": re.findall(r"\bPO-\d{4}\b", text),
        "order_ids": re.findall(r"\bSO-\d{5}\b", text),
        "skus": {p["sku"] for p in products if p["product"].lower() in low},
        "window": next((w for phrase, w in TIME_WINDOWS.items()
                        if phrase in low), None),
        "email_domain": ticket["submitter_email"].split("@")[-1].lower(),
        "contact_sim": {c["contact_id"]: best_window_similarity(tokens, norm(c["name"]))
                        for c in contacts},
        "company_sim": {},
    }
    entered = norm(ticket["company_entered"])
    for a in accounts:
        target = norm(a["company"])
        sim = best_window_similarity(tokens, target)
        if entered:
            sim = max(sim, similarity(entered, target))
            if len(entered.split()) >= 2 and target.startswith(entered):
                sim = max(sim, 0.95)  # "Red Rock" for "Red Rock Rentals"
        hints["company_sim"][a["account_id"]] = sim
    return hints


# ----------------------------------------------------------------------
# Candidates and features
# ----------------------------------------------------------------------
@dataclass
class Candidate:
    order_id: str
    features: dict  # feature name -> number; zeros are simply left out


@dataclass
class Case:
    ticket: dict
    hints: dict
    candidates: list  # every order worth considering, plus NONE at the end


def order_features(order, hints, accounts, created):
    age = (created - date.fromisoformat(order["order_date"])).days
    if age < 0:
        return None  # can't complain about an order placed after the ticket
    account = accounts[order["account_id"]]
    feats = {
        "po_match": float(bool(order["po_number"])
                          and order["po_number"] in hints["po_numbers"]),
        "order_id_match": float(order["order_id"] in hints["order_ids"]),
        "product_match": float(order["sku"] in hints["skus"]),
        "buyer_name_match": scaled(hints["contact_sim"][order["ordered_by_contact_id"]], 0.75),
        "company_match": scaled(hints["company_sim"][order["account_id"]], 0.80),
        "email_domain_match": float(hints["email_domain"] == account["email_domain"]),
    }
    # Blocking: only keep orders with at least one real identity clue. The date
    # alone isn't enough, or half the table would qualify.
    if not any(feats.values()):
        return None
    feats["date_fit"] = date_fit(age, hints["window"])
    return {name: x for name, x in feats.items() if x}


def build_cases(data):
    accounts = {a["account_id"]: a for a in data["accounts"]}
    cases = []
    for ticket in data["tickets"]:
        hints = read_ticket(ticket, data["accounts"], data["contacts"], data["products"])
        created = date.fromisoformat(ticket["created_date"])
        candidates = []
        for order in data["orders"]:
            feats = order_features(order, hints, accounts, created)
            if feats is not None:
                candidates.append(Candidate(order["order_id"], feats))
        quoted = bool(hints["po_numbers"] or hints["order_ids"])
        found = any("po_match" in c.features or "order_id_match" in c.features
                    for c in candidates)
        none_feats = {"none_bias": 1.0}
        if quoted and not found:
            none_feats["unmatched_ref"] = 1.0
        candidates.append(Candidate(NONE, none_feats))
        cases.append(Case(ticket, hints, candidates))
    return cases


# ----------------------------------------------------------------------
# Scoring: weighted sum -> softmax (the same softmax idea as microgpt.py)
# ----------------------------------------------------------------------
def probabilities(case, weights):
    """Return [(order_id, probability), ...], best first."""
    logits = [sum(weights[n] * x for n, x in c.features.items())
              for c in case.candidates]
    top = max(logits)  # subtract the max so exp() can't overflow
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    ranked = [(c.order_id, e / total) for c, e in zip(case.candidates, exps)]
    ranked.sort(key=lambda pair: -pair[1])
    return ranked


def explain(case, order_id, weights, limit=3):
    """Plain-language reasons: the biggest score contributions."""
    cand = next(c for c in case.candidates if c.order_id == order_id)
    parts = sorted(((weights[n] * x, n) for n, x in cand.features.items()),
                   reverse=True)
    return "; ".join(f"{FEATURE_LABELS[n]} ({pts:+.1f})"
                     for pts, n in parts[:limit]) or "no matching clues"


def target_index(case, key):
    """Which candidate SHOULD get the probability? The true order, or NONE
    when there is no true order or it never made the candidate list (in both
    situations the right move is to abstain and ask for more information)."""
    true = key[case.ticket["ticket_id"]]["true_order_id"]
    ids = [c.order_id for c in case.candidates]
    return ids.index(true) if true in ids else ids.index(NONE)
