"""
arrange_data.py  --  Step 1 of the five A's: ARRANGE

Builds a small, fake Acme world so we have something to test against:

    accounts, contacts, products, orders  -> what Acme's systems "know"
    tickets                               -> what the web form gives us
    answer_key                            -> the truth, hidden from the matcher

The matcher we build in the Act step will only see tickets.csv and the
four "system" files. answer_key.csv is used in the Assert step to grade it.

Only the standard library is needed. Run it as many times as you like:
the seed below makes the data identical every run.
"""
import csv
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

SEED = 42
TODAY = date(2026, 11, 12)
OUT_DIR = Path(__file__).parent / "data"

rng = random.Random(SEED)  # our own generator, so results are repeatable


# ----------------------------------------------------------------------
# 1. Reference data: products, companies, people
# ----------------------------------------------------------------------
PRODUCTS = [
    ("ACM-1001", "Rocket Skates"),
    ("ACM-1002", "Giant Magnet"),
    ("ACM-1003", "Anvil (500 lb)"),
    ("ACM-1004", "Dehydrated Boulders"),
    ("ACM-1005", "Jet-Propelled Pogo Stick"),
    ("ACM-1006", "Earthquake Pills"),
    ("ACM-1007", "Instant Tunnel Kit"),
    ("ACM-1008", "Spring-Loaded Catapult"),
    ("ACM-1009", "Giant Rubber Band"),
    ("ACM-1010", "Bird Seed, Industrial"),
]

# (company, email domain, city). ".example" is a reserved domain that can
# never belong to a real company, so none of this can hit a real inbox.
COMPANIES = [
    ("Coyote Pursuit Enterprises", "coyotepursuit.example", "Tucson"),
    ("Mesa Hardware Co.", "mesahardware.example", "Mesa"),
    ("Canyon Outfitters", "canyonoutfitters.example", "Flagstaff"),
    ("Dusty Trails Trading", "dustytrails.example", "Kingman"),
    ("Saguaro Sporting Goods", "saguarosports.example", "Tempe"),
    ("Tumbleweed Traders", "tumbleweedtraders.example", "Yuma"),
    ("Red Rock Rentals", "redrockrentals.example", "Sedona"),
    ("Sidewinder Supply", "sidewindersupply.example", "Phoenix"),
]

FIRST = ["Maria", "Tom", "Priya", "Devon", "Luis", "Hannah", "Omar", "Grace",
         "Ken", "Sofia", "Marcus", "Ava", "Raj", "Nina", "Carl", "Dana",
         "Elena", "Victor", "Mia", "Jonas"]
LAST = ["Nguyen", "Brooks", "Patel", "Okafor", "Ramirez", "Fischer", "Haddad",
        "Kim", "Walsh", "Silva", "Moreno", "Larsen", "Ito", "Greene",
        "Duarte", "Pruitt", "Castillo", "Bergstrom", "Yilmaz", "Abbott"]

# Shuffled once, then we pop from it so no two people share a name.
name_pool = [f"{f} {l}" for f in FIRST for l in LAST]
rng.shuffle(name_pool)

FREE_MAIL = ["mailbox.example", "webmail.example"]


def new_person():
    return name_pool.pop()


def make_phone():
    return f"(602) 555-01{rng.randint(0, 99):02d}"  # 555-01xx is fictional


def make_email(name, domain):
    parts = name.lower().split()
    return f"{parts[0]}.{parts[-1]}@{domain}"


# ----------------------------------------------------------------------
# 2. Accounts and contacts
# ----------------------------------------------------------------------
accounts, contacts = [], []

for n, (company, domain, city) in enumerate(COMPANIES, start=1):
    account_id = f"A-{n:03d}"
    accounts.append({"account_id": account_id, "company": company,
                     "city": city, "email_domain": domain})

    if n == 1:  # our star customer
        people = [("Wile E. Coyote", "buyer"),
                  (new_person(), "accounts payable")]
    else:
        roles = ["buyer"] + rng.sample(
            ["buyer", "accounts payable", "warehouse"], rng.randint(1, 2))
        people = [(new_person(), role) for role in roles]

    for name, role in people:
        contacts.append({
            "contact_id": f"C-{len(contacts) + 1:04d}",
            "account_id": account_id,
            "name": name,
            "role": role,
            "email": make_email(name, domain),
            "phone": make_phone(),
        })

account_by_id = {a["account_id"]: a for a in accounts}
contact_by_id = {c["contact_id"]: c for c in contacts}
product_name = dict(PRODUCTS)
buyers = {a["account_id"]: [c for c in contacts
                            if c["account_id"] == a["account_id"]
                            and c["role"] == "buyer"] for a in accounts}


# ----------------------------------------------------------------------
# 3. Orders (one product line per order, to keep the tutorial simple)
# ----------------------------------------------------------------------
raw_orders = []
used_pos = set()


def add_order(account_id, buyer, days_ago, sku, qty, po, tag=""):
    if po:
        used_pos.add(po)
    raw_orders.append({
        "account_id": account_id,
        "ordered_by_contact_id": buyer["contact_id"],
        "order_date": TODAY - timedelta(days=days_ago),
        "sku": sku, "qty": qty, "po_number": po,
        "ship_to_city": account_by_id[account_id]["city"],
        "tag": tag,
    })


# Planted on purpose: Coyote has TWO Rocket Skates orders, so "the skates he
# ordered last month" only works if the matcher also looks at the date.
coyote = buyers["A-001"][0]
add_order("A-001", coyote, 31, "ACM-1001", 2, "PO-5120", tag="hero")
add_order("A-001", coyote, 95, "ACM-1001", 1, "PO-4388", tag="decoy")

for account in accounts:
    is_coyote = account["account_id"] == "A-001"
    # Coyote's random orders skip Rocket Skates so the two planted ones stay
    # the only candidates for Jane's "the skates he ordered last month".
    choices = PRODUCTS[1:] if is_coyote else PRODUCTS
    for _ in range(rng.randint(40, 60)):
        po = ""
        if rng.random() > 0.25:  # about a quarter of orders have no PO
            while not po or po in used_pos:
                po = f"PO-{rng.randint(1000, 5999)}"
        add_order(
            account["account_id"],
            rng.choice(buyers[account["account_id"]]),
            rng.randint(2, 240),
            rng.choice(choices)[0],
            rng.randint(1, 3) if is_coyote else rng.randint(1, 24),
            po,
        )

raw_orders.sort(key=lambda o: o["order_date"])
orders = []
for i, o in enumerate(raw_orders, start=10001):
    o["order_id"] = f"SO-{i}"
    orders.append({
        "order_id": o["order_id"],
        "order_date": o["order_date"].isoformat(),
        "account_id": o["account_id"],
        "ordered_by_contact_id": o["ordered_by_contact_id"],
        "sku": o["sku"],
        "qty": o["qty"],
        "po_number": o["po_number"],
        "ship_to_city": o["ship_to_city"],
    })

hero_order_id = next(o["order_id"] for o in raw_orders if o["tag"] == "hero")
order_by_id = {o["order_id"]: o for o in orders}


# ----------------------------------------------------------------------
# 4. Tickets: what the web form actually gives us
# ----------------------------------------------------------------------
ISSUES = [
    ("damaged", "arrived damaged"),
    ("damaged", "was crushed in shipping"),
    ("defective", "doesn't work as advertised"),
    ("defective", "failed on first use"),
    ("wrong_item", "isn't what we ordered"),
    ("return", "needs to be sent back"),
]
ASKS = ["a replacement", "a refund", "return instructions", "a call back"]
GENERIC_RELATIONS = [
    "I'm handling this for the person who placed the order.",
    "I'm following up for a colleague who is out.",
    "",
]
DIFFICULTY = {"po": "easy", "order_id": "easy", "name_product": "medium",
              "company_product": "medium", "vague": "hard",
              "no_match": "unmatchable"}


def time_phrase(days):
    if days <= 10:
        return "last week"
    if days <= 45:
        return "last month"
    if days <= 120:
        return "a few months ago"
    return "earlier this year"


def typo(word):
    i = rng.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]  # drop one letter


def build_ticket(order, scenario):
    """Return (what the form captured, what the answer key records)."""
    account = account_by_id[order["account_id"]]
    buyer = contact_by_id[order["ordered_by_contact_id"]]
    product = product_name[order["sku"]]
    order_date = date.fromisoformat(order["order_date"])
    created = min(TODAY, order_date + timedelta(days=rng.randint(4, 40)))
    when = time_phrase((created - order_date).days)

    # Who is filling out the form? Usually 40% a known contact, 60% an
    # outsider. "name_product" tickets are the Jane Doe type, so always outsiders.
    known = rng.random() < 0.4 and scenario != "name_product"
    if known:
        sub = rng.choice([c for c in contacts
                          if c["account_id"] == account["account_id"]])
        name, email, phone = sub["name"], sub["email"], sub["phone"]
        company_entered = account["company"] if rng.random() < 0.8 else ""
        intro = f"Hi, this is {name} from {account['company']}."
    else:
        name = new_person()
        email = make_email(name, rng.choice(FREE_MAIL))
        phone = make_phone()
        short = " ".join(account["company"].split()[:2])
        company_entered = short if rng.random() < 0.3 else ""
        intro = f"Hello, my name is {name}."

    if scenario == "po" and not order["po_number"]:
        scenario = "company_product"  # no PO on file to quote

    issue_type, issue = rng.choice(ISSUES)
    ask = rng.choice(ASKS)
    true_order_id = order["order_id"]

    if scenario == "po":
        body = (f"The {product} on {order['po_number']} {issue}. "
                f"Can we get {ask}?")
    elif scenario == "order_id":
        body = (f"This is about order {order['order_id']}: the {product} "
                f"{issue}. Can we get {ask}?")
    elif scenario == "name_product":
        parts = buyer["name"].split()
        if rng.random() < 0.25 and len(parts[-1]) >= 4:
            parts[-1] = typo(parts[-1])  # misspelled buyer name
        buyer_ref = " ".join(parts)
        intro += f" I'm helping out while {buyer_ref} is unavailable."
        body = (f"The {product} that {buyer_ref} ordered {when} {issue}. "
                f"Can we get {ask}?")
    elif scenario == "company_product":
        body = f"The {product} we ordered {when} {issue}. Can we get {ask}?"
    elif scenario == "vague":
        body = f"Our shipment {issue}. Can someone call me back?"
    else:  # no_match: quotes a PO that doesn't exist in our system
        body = (f"PO-{rng.randint(7000, 7999)}: the {product} {issue}. "
                f"Can we get {ask}?")
        true_order_id = ""

    if not known and scenario != "name_product":
        intro = f"{intro} {rng.choice(GENERIC_RELATIONS)}".strip()

    ticket = {
        "created_date": created.isoformat(),
        "submitter_name": name,
        "submitter_email": email,
        "submitter_phone": phone,
        "company_entered": company_entered,
        "message": f"{intro} {body}",
    }
    key = {
        "true_order_id": true_order_id,
        "true_account_id": account["account_id"],
        "scenario": scenario,
        "difficulty": DIFFICULTY[scenario],
        "issue_type": issue_type,
        "submitter_is_known_contact": known,
    }
    return ticket, key


# T-0001 is our worked example: Jane Doe, writing for the hospitalized buyer.
tickets = [{
    "ticket_id": "T-0001",
    "created_date": TODAY.isoformat(),
    "submitter_name": "Jane Doe",
    "submitter_email": "jane.doe@webmail.example",
    "submitter_phone": "(602) 555-0142",
    "company_entered": "",
    "message": ("Hello, my name is Jane Doe. I'm contacting you on behalf of "
                "Wile E. Coyote, who is in the hospital recovering. The "
                "Rocket Skates he ordered last month arrived damaged. "
                "Could you please send a replacement?"),
}]
answer_key = [{
    "ticket_id": "T-0001",
    "true_order_id": hero_order_id,
    "true_account_id": "A-001",
    "scenario": "hero_jane_doe",
    "difficulty": "medium",
    "issue_type": "damaged",
    "submitter_is_known_contact": False,
}]

# 240 more tickets with a controlled mix of scenarios (enough to train on
# some and hold the rest back for testing).
scenarios = (["po"] * 36 + ["order_id"] * 24 + ["name_product"] * 72 +
             ["company_product"] * 60 + ["vague"] * 30 + ["no_match"] * 18)
rng.shuffle(scenarios)
eligible = [o for o in orders
            if (TODAY - date.fromisoformat(o["order_date"])).days >= 5
            and o["order_id"] != hero_order_id]
chosen = rng.sample(eligible, len(scenarios))

built = [build_ticket(o, s) for o, s in zip(chosen, scenarios)]
built.sort(key=lambda pair: pair[0]["created_date"])
for i, (ticket, key) in enumerate(built, start=2):
    ticket_id = f"T-{i:04d}"
    tickets.append({"ticket_id": ticket_id, **ticket})
    answer_key.append({"ticket_id": ticket_id, **key})


# ----------------------------------------------------------------------
# 5. Save everything
# ----------------------------------------------------------------------
def write_csv(filename, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_csv("accounts.csv", accounts)
write_csv("contacts.csv", contacts)
write_csv("products.csv", [{"sku": s, "product": p} for s, p in PRODUCTS])
write_csv("orders.csv", orders)
write_csv("tickets.csv", tickets)
write_csv("answer_key.csv", answer_key)

print(f"Wrote files to {OUT_DIR}")
print(f"  accounts: {len(accounts)}   contacts: {len(contacts)}   "
      f"orders: {len(orders)}   tickets: {len(tickets)}")
print("\nTickets by scenario:")
for scenario, count in Counter(k["scenario"] for k in answer_key).most_common():
    print(f"  {scenario:<16}{count}")
known_count = sum(k["submitter_is_known_contact"] for k in answer_key)
print(f"\nSubmitter is a known contact on the account: {known_count} of "
      f"{len(answer_key)}")

print("\nWorked example (T-0001):")
print("  ", tickets[0]["message"])
print("   -> true order:", hero_order_id)
skates = [o for o in orders if o["account_id"] == "A-001"
          and o["sku"] == "ACM-1001"]
print("   Coyote Rocket Skates orders (the decoy problem):",
      ", ".join(f"{o['order_id']} ({o['order_date']})" for o in skates))
