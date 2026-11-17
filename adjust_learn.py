"""
adjust_learn.py -- Step 4 of the five A's: ADJUST

Assert showed our hand-set weights were too timid: right about half the time on
"medium" tickets but only ~13% confident. Instead of nudging numbers by hand,
let the computer find better ones by gradient descent. This is the same recipe
as microgpt.py, just with a much smaller model:

    microgpt.py                         this script
    -----------                         -----------
    logits for the next character   ->  a score for each candidate order
    softmax -> probabilities        ->  softmax -> probabilities
    loss = -log(p of the right char)->  loss = -log(p of the right order)
    Value class (autograd)          ->  Value class (a trimmed copy)
    Adam optimizer                  ->  Adam optimizer
    ~thousands of parameters        ->  9 parameters (one weight per feature)

Because we are only learning 9 numbers, you can read them afterwards and ask
"does that make sense?" -- something you can't do with a transformer.

Training uses ONLY the training tickets. Assert grades on the held-out ones.

Run:  python adjust_learn.py
"""
import math
import random

import assert_check
from act_match import HAND_WEIGHTS, run_act
from acme_common import (FEATURES, build_cases, load_answer_key,
                         load_system_data, probabilities, save_weights,
                         split_ticket_ids, target_index)

STEPS = 1500         # how many weight updates
BATCH = 16           # tickets per update
LEARNING_RATE = 0.3
# A small penalty on big weights. This one needed tuning (an Adjust of its own):
# at 0.01 the weights stayed small and the model stayed timid (about 13-50%
# confident even on easy tickets); at 0.0003 it can be sure when the clues are
# strong. Try 0.01 yourself and compare the Assert report.
L2 = 0.0003


# ----------------------------------------------------------------------
# A trimmed-down autograd Value: the same idea as in microgpt.py.
# Each Value remembers its inputs and the local derivative to each one, so
# backward() can walk the graph in reverse applying the chain rule.
# ----------------------------------------------------------------------
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self):
        order, seen = [], set()

        def visit(v):
            if v not in seen:
                seen.add(v)
                for child in v._children:
                    visit(child)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            for child, local in zip(v._children, v._local_grads):
                child.grad += local * v.grad

    __radd__ = __add__
    __rmul__ = __mul__


def ticket_loss(case, target, weights):
    """-log(probability the model gave the right choice) for one ticket."""
    logits = []
    for cand in case.candidates:
        terms = [weights[name] * x for name, x in cand.features.items()]
        logits.append(sum(terms, Value(0.0)))
    shift = max(l.data for l in logits)          # same overflow trick as before
    total = sum(((l - shift).exp() for l in logits), Value(0.0))
    return total.log() - (logits[target] - shift)


def average_loss(cases, targets, weights):
    """The same loss with plain numbers (no autograd), for reporting."""
    total = 0.0
    for case, target in zip(cases, targets):
        p = dict(probabilities(case, weights))[case.candidates[target].order_id]
        total += -math.log(max(p, 1e-12))
    return total / len(cases)


def main():
    data = load_system_data()
    key = load_answer_key()
    cases = build_cases(data)
    train_ids, test_ids = split_ticket_ids(data["tickets"])

    train = [c for c in cases if c.ticket["ticket_id"] in train_ids]
    test = [c for c in cases if c.ticket["ticket_id"] in test_ids]
    train_targets = [target_index(c, key) for c in train]
    test_targets = [target_index(c, key) for c in test]
    print(f"Training on {len(train)} tickets, holding back {len(test)} for testing.")

    weights = {name: Value(0.0) for name in FEATURES}   # start knowing nothing
    params = list(weights.values())
    m = [0.0] * len(params)                              # Adam's running averages
    v = [0.0] * len(params)
    beta1, beta2, eps = 0.9, 0.99, 1e-8
    rng = random.Random(1)
    recent = []

    for step in range(STEPS):
        batch = rng.sample(range(len(train)), BATCH)
        loss = Value(0.0)
        for i in batch:
            loss = loss + ticket_loss(train[i], train_targets[i], weights)
        loss = loss * (1.0 / BATCH)
        loss = loss + sum((w * w for w in params), Value(0.0)) * L2

        loss.backward()

        lr = LEARNING_RATE * (1 - step / STEPS)          # shrink steps over time
        for j, p in enumerate(params):
            m[j] = beta1 * m[j] + (1 - beta1) * p.grad
            v[j] = beta2 * v[j] + (1 - beta2) * p.grad ** 2
            m_hat = m[j] / (1 - beta1 ** (step + 1))
            v_hat = v[j] / (1 - beta2 ** (step + 1))
            p.data -= lr * m_hat / (v_hat ** 0.5 + eps)
            p.grad = 0.0

        recent.append(loss.data)
        if (step + 1) % 150 == 0:
            # One step only sees 16 random tickets, so its loss bounces around.
            # The average of the last 150 steps shows the real trend.
            print(f"  step {step + 1:>4}/{STEPS}   average loss {sum(recent) / len(recent):.4f}")
            recent = []

    learned = {name: w.data for name, w in weights.items()}
    save_weights(learned, "weights_learned.json")

    print("\nAverage loss (lower is better; 0 would mean always 100% sure and right):")
    print(f"  {'':<28}{'train':>8}{'test':>8}")
    print(f"  {'hand-set weights':<28}{average_loss(train, train_targets, HAND_WEIGHTS):>8.3f}"
          f"{average_loss(test, test_targets, HAND_WEIGHTS):>8.3f}")
    print(f"  {'learned weights':<28}{average_loss(train, train_targets, learned):>8.3f}"
          f"{average_loss(test, test_targets, learned):>8.3f}")

    print("\nWeights: what we guessed vs what the data says")
    print(f"  {'feature':<22}{'hand-set':>10}{'learned':>10}")
    for name in FEATURES:
        print(f"  {name:<22}{HAND_WEIGHTS[name]:>10.2f}{learned[name]:>10.2f}")

    run_act(learned, "suggestions_learned.csv")
    print("\nRe-ran Act with the learned weights. Now Assert again:")
    assert_check.main()
    print("\nNext: python answer_queue.py")


if __name__ == "__main__":
    main()
