"""
answer_queue.py -- Step 5 of the five A's: ANSWER

The loop (Act, Assert, Adjust) gave us a matcher we understand and have graded.
Answer is where it meets the CS team: what do they actually see, and who
decides what?

This is where "human in the loop" becomes code. The model never closes a
ticket or sends a refund. It only decides how much help the rep needs:

  confirm_suggestion    very confident AND the sender is a known contact on
                        that account  -> rep confirms with one click
  verify_requester      very confident about the order, but the sender is NOT a
                        known contact (Jane Doe!) -> a person must decide
                        whether this requester is allowed to act on the account
  pick_from_candidates  plausible but not certain -> rep picks from the top 3
  ask_for_info          can't link it -> draft questions for the rep to send

The thresholds come from the Assert report ("if it only acts above X, how
often is it right?"), not from guesswork.

Run:  python answer_queue.py
"""
import csv
from collections import Counter, defaultdict

from acme_common import (AUTO_THRESHOLD, DATA_DIR, NONE, REVIEW_THRESHOLD,
                         build_cases, explain, load_answer_key, load_system_data,
                         load_weights, probabilities, split_ticket_ids)
from act_match import HAND_WEIGHTS


def sender_is_known(ticket, account_id, contacts):
    emails = {c["email"].lower() for c in contacts if c["account_id"] == account_id}
    return ticket["submitter_email"].lower() in emails


def route(ranked, ticket, order_account, contacts):
    top_id, p = ranked[0]
    if top_id == NONE or p < REVIEW_THRESHOLD:
        return "ask_for_info"
    if p >= AUTO_THRESHOLD:
        if sender_is_known(ticket, order_account[top_id], contacts):
            return "confirm_suggestion"
        return "verify_requester"
    return "pick_from_candidates"


def draft_questions(case):
    """Plain templates for now. A language model could write warmer, tailored
    drafts later, but the rep still reads and sends them."""
    hints = case.hints
    asks = []
    if not (hints["po_numbers"] or hints["order_ids"]):
        asks.append("the PO number or order number, if you have it")
    if not hints["skus"]:
        asks.append("which product had the problem")
    asks.append("the name of the person or company that placed the order")
    asks.append("your relationship to that account (so we can confirm we can help you)")
    return ("Hi, thanks for reaching out. To find the right order, could you "
            "send us: " + "; ".join(asks) + "?")


def main():
    data = load_system_data()
    weights = load_weights("weights_learned.json")
    if weights is None:
        print("No learned weights yet (run adjust_learn.py), using the hand-set ones.")
        weights = HAND_WEIGHTS
    cases = build_cases(data)
    order_account = {o["order_id"]: o["account_id"] for o in data["orders"]}

    queue = []
    for case in cases:
        ranked = probabilities(case, weights)
        bucket = route(ranked, case.ticket, order_account, data["contacts"])
        top = [(o, p) for o, p in ranked if o != NONE][:3]
        note = ""
        if bucket == "verify_requester":
            note = (f"Sender {case.ticket['submitter_email']} is not a known contact on "
                    f"{order_account[ranked[0][0]]}. Confirm they may act for this account.")
        elif bucket == "ask_for_info":
            note = draft_questions(case)
        queue.append({
            "ticket_id": case.ticket["ticket_id"],
            "bucket": bucket,
            "suggested_order": top[0][0] if top else "",
            "confidence": f"{top[0][1]:.2f}" if top else "",  # best real order
            "other_candidates": "; ".join(f"{o} ({p:.0%})" for o, p in top[1:]),
            "why": explain(case, ranked[0][0], weights),
            "note_for_rep": note,
        })

    path = DATA_DIR / "review_queue.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0].keys()))
        writer.writeheader()
        writer.writerows(queue)

    counts = Counter(row["bucket"] for row in queue)
    print(f"Routed {len(queue)} tickets. Saved the rep's queue to {path}\n")
    for bucket in ("confirm_suggestion", "verify_requester",
                   "pick_from_candidates", "ask_for_info"):
        print(f"  {bucket:<22}{counts[bucket]:>4}   ({counts[bucket] / len(queue):.0%})")

    # Jane Doe: the case we started with.
    jane = next(row for row in queue if row["ticket_id"] == "T-0001")
    print("\nT-0001 (Jane Doe, writing for Wile E. Coyote)")
    print(f"  Bucket:      {jane['bucket']}")
    print(f"  Suggestion:  {jane['suggested_order']} at {float(jane['confidence']):.0%} confidence")
    print(f"  Why:         {jane['why']}")
    if jane["note_for_rep"]:
        print(f"  Note:        {jane['note_for_rep']}")

    # How trustworthy was each bucket? Only held-out tickets, so it's an honest
    # check (the model never saw these).
    key = load_answer_key()
    _, test_ids = split_ticket_ids(data["tickets"])
    stats = defaultdict(lambda: [0, 0, 0])  # tickets, top suggestion right, true order in top 3
    for case, row in zip(cases, queue):
        if row["ticket_id"] not in test_ids:
            continue
        truth = key[row["ticket_id"]]["true_order_id"]
        top3 = [o for o, _ in probabilities(case, weights)[:3]]
        s = stats[row["bucket"]]
        s[0] += 1
        s[1] += truth != "" and row["suggested_order"] == truth
        s[2] += truth != "" and truth in top3
    print("\nHow each bucket held up on the held-out tickets:")
    print(f"  {'bucket':<22}{'tickets':>8}{'top suggestion right':>24}{'true order in top 3':>22}")
    for bucket in ("confirm_suggestion", "verify_requester",
                   "pick_from_candidates", "ask_for_info"):
        n, right, in3 = stats[bucket]
        if bucket == "ask_for_info":
            print(f"  {bucket:<22}{n:>8}{'(asks for more info)':>24}{'':>22}")
        else:
            print(f"  {bucket:<22}{n:>8}{right:>20}/{n:<3}{in3:>18}/{n}")


if __name__ == "__main__":
    main()
