<div align="center">
  <table border="1">
    <tr>
      <td align="center" style="padding: 20px;">
        <h3>📢 Domain & Email Migration Notice</h3>
        <p>From <b>July 29 th, 2026</b>, Cakradana will transition to new domains as <code>cakradana.org</code> will not be renewed:</p>
        <p>🌐 <b>Website:</b> <a href="https://cakradana.faizath.com">cakradana.faizath.com</a> (formerly <i>cakradana.org</i>)<br>
        ⚙️ <b>API:</b> <a href="https://cakradana-api.faizath.com">cakradana-api.faizath.com</a> (formerly <i>api.cakradana.org</i>)<br>
        📧 <b>Email:</b> <a href="mailto:contact@cakradana.faizath.com">contact@cakradana.faizath.com</a> (formerly <i>contact@cakradana.org</i>)<br>
        🛰️ <b>CDN:</b> <a>cakradana-cdn.faizath.com</a> (formerly <i>cdn.cakradana.org</i>)<br>
        📈 <b>Status Pages:</b> <a href="https://status.faizath.com/status/cakradana">https://status.faizath.com/status/cakradana</a> (formerly <i>status.cakradana.org</i>)
        </p>
      </td>
    </tr>
  </table>
</div>

# 🏛️ Cakradana AI

<div align="center">

![Cakradana Logo](assets/logo.png)

**AI System for Transparency in Indonesian Election Financing**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-orange.svg)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-red.svg)](https://scikit-learn.org/)

</div>

---

Detects donations that warrant human investigation, explains why each one was
surfaced, and improves as investigators report what they found.

The output is a prioritised queue, not a verdict. Nothing here determines that
an offence occurred.

## What it produces

Two independent verdicts, never combined into one number.

**Legal findings** are deterministic tests of statutory compliance, each with
the article it rests on, the threshold applied, and the value observed. Whether
a donation exceeds a limit is arithmetic; a probability would be a worse answer
to a question that has an exact one.

**A behavioural score** from 0 to 100 ranks everything the statute has already
cleared, with the factors that drove it. It is an estimate about conduct and
says nothing about guilt.

Blending the two would produce a quantity that is neither auditable as a legal
finding nor interpretable as a probability, and that could not be defended when
its subject objects to it.

## The two-tier rule engine

**Tier 1 — statutory.** Eleven rules covering donation limits, cumulative
limits, prohibited sources, and reporting compliance, drawn from UU Parpol
(UU No. 2/2011 amending UU No. 2/2008) and UU Pemilu (UU No. 7/2017) with
PKPU No. 18/2023 as implementing regulation.

The cumulative rules matter most. The limits are per donor per period, not per
transaction, so twenty donations of Rp200,000,000 each breach the annual party
limit twentyfold while every individual payment passes a single-transaction
check. Nothing in this system detected that before.

**Tier 2 — behavioural.** Ten heuristics for fan-in convergence, pass-through
routing, amounts positioned below a limit, velocity spikes, and deadline
clustering. These carry no legal basis and cite none. Their output is a
training label and a ranking signal, and they are explicitly fallible.

Keeping the tiers apart is what lets a classifier contribute anything. Trained
on statutory outcomes it could only relearn arithmetic it was already given,
and applied to donations the statute had cleared it would return negatives by
construction.

### Rules are data

Rules live in `cakradana/rules/rulesets/` as versioned YAML with effective
dates. Amending a threshold needs no code change and no retrain, and a donation
is always evaluated against the rules in force on its own date.

A rule that cannot evaluate its inputs returns `INDETERMINATE`, never `PASS`.
An unevaluated prohibition reported as clean is indistinguishable from a real
clean result, which makes it worse than no answer at all.

**No statutory citation in this repository has been verified by a legal
reviewer.** The thresholds and articles are consistent across the project's
source documents, and consistency is not verification. Until each is checked
against the consolidated text, the engine reports Tier-1 rules as indeterminate
rather than asserting an unverified legal fact about a named person. Set
`require_verified_citations=False` only against synthetic data.

## Detection lanes

| Lane | Share of the score | Basis |
|---|---|---|
| Classifier | 50% | LightGBM over the full feature set |
| Graph | 30% | Structural findings — convergence, pass-through, concentration |
| Anomaly | 15% | Isolation Forest over donations the rules cleared |
| Reputation | 5% | Not operating |

Each lane is capped separately. They are not calibrated against one another,
and the exploratory ones can always surface more unusual donations than a team
can review, so an unbounded pool lets the weakest evidence displace the
strongest.

A lane that cannot run says so, and the score is marked incomplete rather than
having the remaining lanes stretched to cover the gap.

**The reputation lane does not operate.** It would accuse named parties on the
strength of press coverage, and it stays off until its accuracy and defamation
controls are in place.

## How it is measured

Accuracy is not reported. On a realistic population where a few percent of
donations are risky, flagging nothing scores above 95%.

The binding constraint is analyst hours, so metrics are defined against the
number of donations a team can actually review in a period:

- **Precision@B** — of the B donations reviewed, how many were genuinely risky.
- **Recall@B** — of all genuinely risky donations, how many reached the queue.
- **Lift@B** — confirmed-risky donations the model surfaced that **no rule
  flagged**, over what the rules alone would surface at the same budget.

**Lift@B decides whether the classifier ships.** At or below parity the rules
run alone: they are cheaper, already explainable, and already built. A model
trained on heuristic labels can reproduce those heuristics and look capable
while adding nothing, and this is the only measurement that shows it.

Splits are grouped by donor and the absence of overlap is asserted, not
reported. Evaluation uses human-confirmed labels; agreement with the heuristics
is not a success metric.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

Generate a dataset and verify it contains the patterns it labels:

```bash
cakradana generate
```

Train, measure, and get a shipping decision:

```bash
cakradana train --budget 100 --version lgbm-2026.08.1
```

Score a dataset with the rules alone:

```bash
cakradana score
```

Run the service:

```bash
export CAKRADANA_SERVICE_TOKEN=...
uvicorn cakradana.serving.api:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs
```

## The scoring contract

Callers send a canonical donation record. They do **not** send engineered
features, and a request carrying one is rejected rather than obeyed.

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Authorization: Bearer $CAKRADANA_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-1",
    "donation": {
      "donation_id": "don-1",
      "sender_ref":   {"entity_id": "donor-1", "entity_type": "individual"},
      "receiver_ref": {"entity_id": "party-1", "entity_type": "political-party"},
      "amount_idr": 95000000,
      "occurred_at": "2026-08-12T00:00:00+07:00",
      "recorded_at": "2026-08-13T04:21:00+07:00",
      "channel": "paper-form",
      "electoral_context": "pemilu-2029"
    }
  }'
```

This is the decision that makes the two services connectable. Features are
derived here from maintained point-in-time state, which is also what keeps
training and serving on one implementation rather than a convention.

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | Score one donation |
| `POST /v1/score/batch` | Bounded batch, reported per item |
| `POST /v1/rescore` | Score again with a reason; the previous result is kept |
| `GET /v1/explain/{donation_id}` | Every scoring event for a donation |
| `GET /v1/rules` | Active rule set, effective dates, verification status |
| `GET /v1/model-info` | Model, rule-set, and feature versions |
| `GET /health`, `GET /ready` | Liveness, distinct from readiness |

Readiness is separate from liveness because a running process with no rules
loaded would report every donation as carrying no findings, which reads exactly
like a clean bill of health.

## Point-in-time correctness

Every aggregate is computed through a view bound to one timestamp. A donation
contributes only when it both occurred and was recorded at or before that
moment.

The second condition is the one that is easy to lose. A donation that occurred
in January but was scraped in June was not knowable in February, and admitting
it to a February aggregate leaks the future into a past decision. Computing
aggregates over a finished dataset before splitting it is the same error at
training time.

Values that cannot be computed are `null`, never `0.0`. A donor's first
donation has no standard deviation and no mean interval between donations;
filling those with zero asserts a perfectly regular donor, which is both false
and a strong signal in the wrong direction.

## Synthetic data

Bootstrap only, and never reported as system performance.

Every typology is generated with the structure that defines it and checked at
generation time by a detector using only that signal. A dataset whose labels
are not supported by its own structure fails to build.

Ordinary giving is generated alongside it, including genuine fundraising
surges. These converge many donors on one recipient exactly as a split
contribution does, and differ in that real supporters choose varied amounts.
Without them any fan-in detector reaches perfect precision here and collapses
on contact with real data.

The risky share is a realistic few percent rather than half. A balanced set
makes class weighting inert and yields precision estimates that do not transfer
to a population where the pattern is rare.

## Layout

```
cakradana/
├── schema/       canonical donation, entity, and label records
├── rules/        two-tier engine; rule sets as versioned YAML
├── features/     one implementation per feature, shared by train and serve
├── lanes/        classifier, graph, anomaly
├── scoring/      composition, bands, reason codes
├── evaluation/   budget metrics, lift, calibration, splits
├── training/     pipeline, artifact registry
├── serving/      scoring service and HTTP contract
├── data/         synthetic generator and its acceptance checks
├── history.py    point-in-time views
├── calendar.py   campaign periods and limit-regime selection
└── registers.py  prohibited-source and conviction registers
```

## What is not built

Stated because absence is easy to mistake for coverage.

| Capability | Blocked on |
|---|---|
| Foreign-source detection | Donor jurisdiction is not captured; nationality must never be inferred from a name |
| Prohibited-source findings | An authoritative register of government bodies, state and regional enterprises, and village governments |
| Proceeds-of-crime findings | A register of convictions with final legal force; press reporting does not meet that standard |
| Report reconciliation | Campaign finance submissions and designated-account transactions |
| Self-funding detection | Records do not distinguish a candidate's own funds from a third party's |
| External reputation | Accuracy and defamation controls |

Each is implemented and wired, reports indeterminate, and states why. The
mechanism is ready the moment the data is.

## Ethics

An analytical decision-support tool. Results are not evidence of a violation.
False positives here damage named individuals in a political context, so the
system is deliberately modest about what it knows: it ranks donations for
human attention and leaves determination to people.
