"""Synthetic donation generator.

Bootstrap data for a system that has no labelled real data yet. Its one hard
obligation is that every typology it claims to produce is actually present in
the structure of the data, not merely written into a label column.

The generator this replaces did not meet that. It drew random amounts, wrote a
typology name beside them, and left the defining structure absent: rows marked
as donation splitting had no convergence of donors, rows marked self-funded
were generated identically to ordinary ones, and the illegal-source signal
lived entirely in a donor's name, which training then discarded. A model cannot
recover a pattern that was never encoded, so most of the reported detection was
measuring noise.

Two properties guard against repeating that.

The base rate is realistic. A half-risky dataset makes class weighting inert
and produces precision estimates that do not survive contact with a real
population.

Every typology is checked at generation time by a detector that uses only its
defining signal. If donation splitting cannot be recovered by counting
converging donors, the structure was not generated, and the generator says so
rather than emitting a dataset that looks fine until a model trains on it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Sequence

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.registers import Register, RegisterEntry, RegisterSet
from cakradana.reporting import ReportedDonation, SubmissionSet
from cakradana.schema import (
    Channel,
    Donation,
    Entity,
    EntityRef,
    EntityType,
    FieldProvenance,
    Provenance,
    TemporalPrecision,
    TransactionKind,
)

WIB = timezone(timedelta(hours=7), name="WIB")

GENERATOR_VERSION = "synthetic-2026.08.1"

INDIVIDUAL_PARTY_LIMIT = 200_000_000
COMPANY_PARTY_LIMIT = 800_000_000

#: Typologies this generator encodes structurally.
T_CUMULATIVE = "T-02"
T_FOREIGN = "T-03"
T_ILLEGAL_SOURCE = "T-05"
T_UNREPORTED = "T-07"
T_SMURFING = "T-09"
T_PROXY = "T-10"
T_SELF_FUNDED = "T-11"
T_STRUCTURING = "T-12"

ALL_TYPOLOGIES = (
    T_CUMULATIVE,
    T_FOREIGN,
    T_ILLEGAL_SOURCE,
    T_UNREPORTED,
    T_SMURFING,
    T_PROXY,
    T_SELF_FUNDED,
    T_STRUCTURING,
)

#: Jurisdictions used for the foreign-source pattern. Carried on the entity,
#: never inferred from a name: name-based nationality inference is unreliable
#: and discriminatory, and here it would attach a statutory offence to someone
#: on the basis of what they are called.
_FOREIGN_JURISDICTIONS = ("SG", "MY", "HK", "AU", "CN")

_GIVEN = (
    "Budi", "Siti", "Agus", "Dewi", "Eko", "Rina", "Joko", "Sri", "Andi", "Ayu",
    "Bambang", "Nur", "Hendra", "Lestari", "Rizki", "Fitri", "Dian", "Wahyu",
    "Putri", "Arif", "Maya", "Teguh", "Indah", "Yusuf", "Ratna",
)
_FAMILY = (
    "Santoso", "Wijaya", "Pratama", "Kusuma", "Halim", "Nugroho", "Saputra",
    "Hidayat", "Permana", "Setiawan", "Gunawan", "Utami", "Firmansyah",
    "Maulana", "Suryani",
)
_COMPANY_STEM = (
    "Sumber Sejahtera", "Karya Mandiri", "Cahaya Nusantara", "Bina Usaha",
    "Mitra Abadi", "Sentosa Jaya", "Tirta Makmur", "Graha Persada",
)
_PROHIBITED = (
    ("PT Perusahaan Listrik Negara (Persero)", "state-enterprise"),
    ("PT Pertamina (Persero)", "state-enterprise"),
    ("Perumda Air Minum Tirta", "regional-enterprise"),
    ("Dinas Pekerjaan Umum Kabupaten", "government"),
    ("Pemerintah Desa Sukamaju", "village-government"),
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Generation parameters. Recorded in the manifest with the output."""

    seed: int = 20260816
    n_legitimate_donors: int = 900
    n_recipients: int = 12
    n_background_donations: int = 6000
    #: Share of donations that belong to a risky pattern. A realistic
    #: prevalence, not a balanced dataset.
    risky_rate: float = 0.03
    period_start: date = date(2026, 1, 1)
    period_end: date = date(2026, 12, 31)
    electoral_context: str = "pemilu-2029"
    #: The declared campaign window, a sub-span of the year rather than the
    #: whole of it. Donations inside it fall under the campaign limits, which
    #: are an order of magnitude higher than the annual party limits governing
    #: the rest of the year. A calendar that marked the entire year as campaign
    #: would judge every donation against the most permissive cap available and
    #: leave the annual limits untested.
    campaign_start: date = date(2026, 9, 1)
    campaign_end: date = date(2026, 11, 24)
    reporting_deadline: date = date(2026, 10, 15)
    #: Recipients that attract genuine grassroots fan-in. Without benign
    #: convergence in the negatives, any fan-in detector scores perfectly here
    #: and fails immediately on real data.
    n_grassroots_campaigns: int = 6
    typology_mix: tuple[tuple[str, float], ...] = (
        (T_SMURFING, 0.24),
        (T_CUMULATIVE, 0.20),
        (T_STRUCTURING, 0.16),
        (T_PROXY, 0.12),
        (T_ILLEGAL_SOURCE, 0.08),
        (T_FOREIGN, 0.08),
        (T_SELF_FUNDED, 0.06),
        (T_UNREPORTED, 0.06),
    )
    #: Share of lawful donations that a recipient omits from its filed return.
    #: Reporting failures are not confined to donations that are otherwise
    #: irregular, and generating them only alongside other patterns would make
    #: the reconciliation rule look like a proxy for the others.
    unreported_share: float = 0.04


@dataclass
class SyntheticDataset:
    """Generated donations with the truth about how they were built."""

    donations: list[Donation]
    entities: dict[str, Entity]
    #: Donation id to typology, for donations that are part of a risky pattern.
    truth: dict[str, str]
    registers: RegisterSet
    calendar: ElectoralCalendar
    #: Filed campaign finance returns covering the generated period. Every
    #: donation appears in them except those marked as unreported, so the
    #: reconciliation rule has something to find and something to leave alone.
    submissions: SubmissionSet
    manifest: dict[str, object]

    def __len__(self) -> int:
        return len(self.donations)

    @property
    def risky_ids(self) -> set[str]:
        return set(self.truth)

    def typology_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {t: 0 for t in ALL_TYPOLOGIES}
        for typology in self.truth.values():
            counts[typology] = counts.get(typology, 0) + 1
        return counts


class _Builder:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.donations: list[Donation] = []
        self.entities: dict[str, Entity] = {}
        self.truth: dict[str, str] = {}
        self._counter = 0

    # -- entity helpers --------------------------------------------------

    def _person_name(self) -> str:
        return f"{self.rng.choice(_GIVEN)} {self.rng.choice(_FAMILY)}"

    def _company_name(self) -> str:
        return f"PT {self.rng.choice(_COMPANY_STEM)} {self.rng.randint(1, 999)}"

    def entity(
        self, entity_id: str, name: str, entity_type: EntityType, **kwargs
    ) -> Entity:
        entity = Entity(
            entity_id=entity_id, canonical_name=name, entity_type=entity_type, **kwargs
        )
        self.entities[entity_id] = entity
        return entity

    def next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:06d}"

    # -- donation helper -------------------------------------------------

    def add(
        self,
        sender: Entity,
        receiver: Entity,
        amount: int,
        when: datetime,
        *,
        typology: str | None = None,
        channel: Channel | None = None,
        recorded: datetime | None = None,
        self_funded: bool | None = False,
    ) -> Donation:
        channel = channel or self.rng.choices(
            [Channel.DIGITAL_FORM, Channel.PAPER_FORM, Channel.WEB_SCRAPE],
            weights=[0.6, 0.25, 0.15],
        )[0]

        # Scanned and scraped records arrive later and less precisely than
        # submitted ones, which is what makes the quality features carry signal
        # and what makes late arrival worth modelling at all.
        if recorded is None:
            lag_days = {
                Channel.DIGITAL_FORM: 0,
                Channel.PAPER_FORM: self.rng.randint(1, 14),
                Channel.WEB_SCRAPE: self.rng.randint(7, 90),
            }[channel]
            recorded = when + timedelta(days=lag_days)

        precision = (
            TemporalPrecision.MINUTE
            if channel is Channel.DIGITAL_FORM
            else TemporalPrecision.DAY
        )
        if precision is TemporalPrecision.DAY:
            when = when.replace(hour=0, minute=0, second=0, microsecond=0)

        provenance = {}
        if channel is Channel.PAPER_FORM:
            provenance["amount_idr"] = FieldProvenance(
                provenance=Provenance.EXTRACTED,
                confidence=round(self.rng.uniform(0.62, 0.98), 3),
                extractor_version="ocr-1",
            )
        elif channel is Channel.WEB_SCRAPE:
            provenance["amount_idr"] = FieldProvenance(
                provenance=Provenance.SCRAPED,
                confidence=round(self.rng.uniform(0.70, 0.99), 3),
            )
        else:
            provenance["amount_idr"] = FieldProvenance(
                provenance=Provenance.SUBMITTED, confidence=1.0
            )

        donation = Donation(
            donation_id=self.next_id("don"),
            sender_ref=EntityRef(
                entity_id=sender.entity_id,
                raw_text=sender.canonical_name,
                entity_type=sender.entity_type,
                resolution_confidence=1.0,
            ),
            receiver_ref=EntityRef(
                entity_id=receiver.entity_id,
                raw_text=receiver.canonical_name,
                entity_type=receiver.entity_type,
                resolution_confidence=1.0,
            ),
            amount_idr=amount,
            occurred_at=when,
            occurred_at_precision=precision,
            recorded_at=recorded,
            transaction_kind=self.rng.choices(
                [TransactionKind.TRANSFER, TransactionKind.CASH],
                weights=[0.8, 0.2],
            )[0],
            channel=channel,
            electoral_context=self.config.electoral_context,
            is_self_funded_declared=self_funded,
            provenance=provenance,
        )
        self.donations.append(donation)
        if typology:
            self.truth[donation.donation_id] = typology
        return donation

    # -- time helper -----------------------------------------------------

    def random_datetime(self) -> datetime:
        """A donation date with a realistic campaign shape.

        Weighted towards the run-up to the reporting deadline, so that
        deadline-clustering is testable and time-based splitting means
        something.
        """
        start, end = self.config.period_start, self.config.period_end
        span = (end - start).days
        deadline_offset = (self.config.reporting_deadline - start).days

        if self.rng.random() < 0.25:
            offset = int(
                min(
                    max(self.rng.gauss(deadline_offset - 5, 12), 0),
                    span,
                )
            )
        else:
            offset = self.rng.randint(0, span)
        return datetime.combine(
            start + timedelta(days=offset), datetime.min.time(), tzinfo=WIB
        ) + timedelta(hours=self.rng.randint(6, 21), minutes=self.rng.randint(0, 59))

    def random_datetime_outside_campaign(self) -> datetime:
        """A date governed by the annual party limits.

        Patterns built to sit against a specific cap are placed where that cap
        applies. The campaign limits are an order of magnitude higher, so a
        contribution shaped to hug the annual limit is unremarkable inside a
        campaign window, and generating it there would label a pattern the data
        does not contain.
        """
        config = self.config
        for _ in range(50):
            when = self.random_datetime()
            if not (config.campaign_start <= when.date() <= config.campaign_end):
                return when
        # Fall back to the start of the year, which is always outside the
        # declared campaign window.
        return datetime.combine(
            config.period_start, datetime.min.time(), tzinfo=WIB
        ) + timedelta(days=self.rng.randint(0, 30))

    def heavy_tailed_amount(self) -> int:
        """A donation amount from a heavy-tailed distribution.

        Real giving is dominated by small amounts with a long upper tail.
        A uniform draw would make ordinary large donations look exceptional and
        every threshold feature meaningless.
        """
        value = math.exp(self.rng.gauss(math.log(3_000_000), 1.15))
        value = min(value, INDIVIDUAL_PARTY_LIMIT * 0.8)
        # Round the way people actually give.
        step = 100_000 if value < 10_000_000 else 1_000_000
        return max(step, int(round(value / step) * step))


def generate(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Build a dataset. Deterministic for a given seed."""
    config = config or GeneratorConfig()
    builder = _Builder(config)

    recipients = [
        builder.entity(
            f"party-{i:03d}",
            f"Partai Contoh {chr(ord('A') + i)}",
            EntityType.POLITICAL_PARTY,
            jurisdiction="ID",
        )
        for i in range(config.n_recipients)
    ]
    donors = [
        builder.entity(
            f"donor-{i:05d}",
            builder._person_name() if i % 5 else builder._company_name(),
            EntityType.INDIVIDUAL if i % 5 else EntityType.CORPORATION,
            # Recorded, not inferred. A donor whose jurisdiction is simply
            # unknown leaves the foreign-source rule unable to answer, which is
            # the honest outcome but tells nobody anything.
            jurisdiction="ID",
        )
        for i in range(config.n_legitimate_donors)
    ]

    _background(builder, donors, recipients)
    _grassroots(builder, recipients)

    target_risky = int(config.n_background_donations * config.risky_rate)
    for typology, share in config.typology_mix:
        _EMITTERS[typology](builder, recipients, int(target_risky * share))

    prohibited = _prohibited_register(builder)
    calendar = ElectoralCalendar(
        [
            CampaignPeriod(
                electoral_context=config.electoral_context,
                start=config.campaign_start,
                end=config.campaign_end,
                reporting_deadlines=(config.reporting_deadline,),
                label=f"campaign {config.campaign_start.year}",
            )
        ]
    )

    builder.donations.sort(key=lambda d: d.occurred_at)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "donations": len(builder.donations),
        "entities": len(builder.entities),
        "risky_donations": len(builder.truth),
        "observed_risky_rate": round(
            len(builder.truth) / max(len(builder.donations), 1), 4
        ),
        "configured_risky_rate": config.risky_rate,
        "typology_counts": {
            t: sum(1 for v in builder.truth.values() if v == t) for t in ALL_TYPOLOGIES
        },
        "period": [str(config.period_start), str(config.period_end)],
    }

    return SyntheticDataset(
        donations=builder.donations,
        entities=builder.entities,
        truth=builder.truth,
        registers=RegisterSet([prohibited, _convictions_register(builder)]),
        calendar=calendar,
        submissions=_submissions(builder, config),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------


def _background(
    builder: _Builder, donors: Sequence[Entity], recipients: Sequence[Entity]
) -> None:
    """Ordinary giving: mostly one-off, some recurring, heavy-tailed amounts."""
    config = builder.config
    rng = builder.rng

    recurring = rng.sample(donors, k=max(1, len(donors) // 6))
    for donor in recurring:
        recipient = rng.choice(recipients)
        n = rng.randint(3, 9)
        first = builder.random_datetime()
        cadence = rng.randint(21, 60)
        for i in range(n):
            when = first + timedelta(days=cadence * i + rng.randint(-3, 3))
            if when.date() > config.period_end:
                break
            builder.add(donor, recipient, builder.heavy_tailed_amount(), when)

    while len(builder.donations) < config.n_background_donations:
        builder.add(
            rng.choice(donors),
            rng.choice(recipients),
            builder.heavy_tailed_amount(),
            builder.random_datetime(),
            self_funded=False,
        )


def _grassroots(builder: _Builder, recipients: Sequence[Entity]) -> None:
    """Genuine fundraising bursts.

    These converge many donors on one recipient in a short window, exactly like
    donation splitting does. They differ in that the amounts vary widely and a
    substantial share of the donors have given before. Without them, a fan-in
    detector reaches perfect precision on synthetic data and collapses on real
    data.
    """
    rng = builder.rng
    for _ in range(builder.config.n_grassroots_campaigns):
        recipient = rng.choice(recipients)
        start = builder.random_datetime()
        supporters = []
        for _ in range(rng.randint(25, 70)):
            donor = builder.entity(
                builder.next_id("supporter"),
                builder._person_name(),
                EntityType.INDIVIDUAL,
                jurisdiction="ID",
            )
            supporters.append(donor)

        # Roughly half arrive with prior history elsewhere, which is what
        # separates a real support base from a cohort created for one purpose.
        # Their earlier donation is kept inside the declared period so that
        # every record falls under a known limit regime.
        floor = datetime.combine(
            builder.config.period_start, datetime.min.time(), tzinfo=WIB
        )
        for donor in supporters[: len(supporters) // 2]:
            earlier = start - timedelta(days=rng.randint(40, 200))
            if earlier < floor:
                earlier = floor + timedelta(hours=rng.randint(0, 72))
            if earlier >= start:
                continue
            builder.add(
                donor,
                rng.choice(recipients),
                builder.heavy_tailed_amount(),
                earlier,
            )

        for donor in supporters:
            builder.add(
                donor,
                recipient,
                builder.heavy_tailed_amount(),
                start + timedelta(hours=rng.randint(0, 240)),
            )


# ---------------------------------------------------------------------------
# Risky patterns — each encoded with the structure that defines it
# ---------------------------------------------------------------------------


def _smurfing(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Many nominal donors, one recipient, tight window, homogeneous amounts.

    The defining structure is convergence plus homogeneity plus donors with no
    other footprint. All three are generated; the label follows from them
    rather than standing in for them.
    """
    rng = builder.rng
    emitted = 0
    while emitted < target:
        recipient = rng.choice(recipients)
        n_donors = rng.randint(16, 35)
        base = rng.choice([5_000_000, 10_000_000, 15_000_000, 20_000_000])
        start = builder.random_datetime()
        # Coordinated splitting is executed quickly. Spreading a cohort across
        # a full fortnight would dilute the convergence that defines it, and
        # would make the pattern indistinguishable from ordinary traffic.
        span_hours = rng.randint(24, 7 * 24)
        for _ in range(n_donors):
            donor = builder.entity(
                builder.next_id("smurf"),
                builder._person_name(),
                EntityType.INDIVIDUAL,
                jurisdiction="ID",
            )
            amount = int(base * rng.uniform(0.96, 1.04) // 100_000 * 100_000)
            builder.add(
                donor,
                recipient,
                max(amount, 100_000),
                start + timedelta(hours=rng.randint(0, span_hours)),
                typology=T_SMURFING,
            )
            emitted += 1


def _proxy(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """An intermediary that receives and forwards, with no other activity."""
    rng = builder.rng
    emitted = 0
    while emitted < target:
        origin = builder.entity(
            builder.next_id("origin"), builder._company_name(), EntityType.CORPORATION,
            jurisdiction="ID",
        )
        intermediary = builder.entity(
            builder.next_id("proxy"), builder._person_name(), EntityType.INDIVIDUAL,
            jurisdiction="ID",
        )
        recipient = rng.choice(recipients)
        amount = rng.randint(50_000_000, 190_000_000)
        inflow_at = builder.random_datetime()

        builder.add(origin, intermediary, amount, inflow_at, typology=T_PROXY)
        emitted += 1
        builder.add(
            intermediary,
            recipient,
            int(amount * rng.uniform(0.88, 1.10)),
            inflow_at + timedelta(days=rng.randint(1, 6)),
            typology=T_PROXY,
        )
        emitted += 1


def _structuring(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Repeated amounts sitting deliberately just under the limit."""
    rng = builder.rng
    emitted = 0
    while emitted < target:
        donor = builder.entity(
            builder.next_id("structurer"), builder._person_name(), EntityType.INDIVIDUAL,
            jurisdiction="ID",
        )
        recipient = rng.choice(recipients)
        start = builder.random_datetime_outside_campaign()
        for i in range(rng.randint(3, 6)):
            ratio = rng.uniform(0.91, 0.985)
            builder.add(
                donor,
                recipient,
                int(INDIVIDUAL_PARTY_LIMIT * ratio // 100_000 * 100_000),
                start + timedelta(days=rng.randint(10, 60) * (i + 1)),
                typology=T_STRUCTURING,
            )
            emitted += 1


def _cumulative(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Individually compliant donations that together breach the period cap."""
    rng = builder.rng
    emitted = 0
    while emitted < target:
        donor = builder.entity(
            builder.next_id("cumulative"), builder._person_name(), EntityType.INDIVIDUAL,
            jurisdiction="ID",
        )
        recipient = rng.choice(recipients)
        n = rng.randint(4, 9)
        # Each donation lawful on its own; the sum is not.
        share = int(INDIVIDUAL_PARTY_LIMIT * rng.uniform(0.28, 0.45))
        start = builder.random_datetime_outside_campaign()
        for i in range(n):
            builder.add(
                donor,
                recipient,
                share,
                start + timedelta(days=rng.randint(5, 40) * (i + 1)),
                typology=T_CUMULATIVE,
            )
            emitted += 1


def _illegal_source(
    builder: _Builder, recipients: Sequence[Entity], target: int
) -> None:
    """Donations from entities that belong to the prohibited-source register.

    The signal is register membership, carried on the entity, and not the
    entity's name. The previous generator encoded it in the name and then
    dropped the name before training, so the only trace of the typology was
    removed from what the model could see.
    """
    rng = builder.rng
    prohibited = [
        builder.entity(
            f"prohibited-{i:03d}",
            name,
            EntityType.STATE_ENTERPRISE
            if category.endswith("enterprise")
            else EntityType.GOVERNMENT,
            registers=("prohibited_source",),
        )
        for i, (name, category) in enumerate(_PROHIBITED)
    ]
    for _ in range(target):
        builder.add(
            rng.choice(prohibited),
            rng.choice(recipients),
            builder.heavy_tailed_amount(),
            builder.random_datetime(),
            typology=T_ILLEGAL_SOURCE,
        )


def _foreign(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Donations from parties whose jurisdiction is recorded as foreign.

    The signal lives on the entity, as a recorded jurisdiction. Nothing about
    the donor's name indicates it, deliberately: a detector that keyed on names
    would be inferring nationality from what someone is called.
    """
    rng = builder.rng
    for _ in range(target):
        donor = builder.entity(
            builder.next_id("foreign"),
            builder._company_name(),
            EntityType.FOREIGN_ENTITY,
            jurisdiction=rng.choice(_FOREIGN_JURISDICTIONS),
        )
        builder.add(
            donor,
            rng.choice(recipients),
            builder.heavy_tailed_amount(),
            builder.random_datetime(),
            typology=T_FOREIGN,
        )


def _self_funded(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Declared self-funding preceded by an unexplained inflow.

    The pattern is a candidate declaring money as their own shortly after
    receiving a comparable sum from somewhere else. Without the declaration
    there is nothing to test, which is why this typology needs the field rather
    than merely the amounts.
    """
    rng = builder.rng
    emitted = 0
    while emitted < target:
        candidate = builder.entity(
            builder.next_id("candidate"), builder._person_name(), EntityType.INDIVIDUAL,
            jurisdiction="ID",
        )
        backer = builder.entity(
            builder.next_id("backer"), builder._company_name(), EntityType.CORPORATION,
            jurisdiction="ID",
        )
        recipient = rng.choice(recipients)
        declared = rng.randint(60_000_000, 180_000_000)
        inflow_at = builder.random_datetime_outside_campaign()

        # The inflow is an ordinary-looking donation to the candidate, recorded
        # promptly. A scraped record arriving months later would not have been
        # knowable when the declaration was scored, and the pattern would be
        # undetectable for reasons that have nothing to do with the pattern.
        builder.add(
            backer,
            candidate,
            int(declared * rng.uniform(0.6, 1.1)),
            inflow_at,
            channel=Channel.DIGITAL_FORM,
        )
        builder.add(
            candidate,
            recipient,
            declared,
            inflow_at + timedelta(days=rng.randint(3, 25)),
            typology=T_SELF_FUNDED,
            self_funded=True,
        )
        emitted += 1


def _unreported(builder: _Builder, recipients: Sequence[Entity], target: int) -> None:
    """Ordinary donations that the recipient will omit from its filed return.

    Generated as unremarkable in every other respect. A reporting failure that
    only ever accompanied another irregularity would make the reconciliation
    rule a proxy for those, and it would never be tested on its own.
    """
    rng = builder.rng
    donors = [
        builder.entity(
            builder.next_id("undeclared"), builder._person_name(), EntityType.INDIVIDUAL,
            jurisdiction="ID",
        )
        for _ in range(max(target, 1))
    ]
    for donor in donors[:target]:
        builder.add(
            donor,
            rng.choice(recipients),
            builder.heavy_tailed_amount(),
            builder.random_datetime(),
            typology=T_UNREPORTED,
        )


_EMITTERS = {
    T_FOREIGN: _foreign,
    T_SELF_FUNDED: _self_funded,
    T_UNREPORTED: _unreported,
    T_SMURFING: _smurfing,
    T_PROXY: _proxy,
    T_STRUCTURING: _structuring,
    T_CUMULATIVE: _cumulative,
    T_ILLEGAL_SOURCE: _illegal_source,
}


def _submissions(builder: _Builder, config: GeneratorConfig) -> SubmissionSet:
    """Build the filings a recipient would have made.

    Everything is declared except the donations generated as unreported, plus a
    random handful of ordinary ones. Reporting failures are not confined to
    donations that are otherwise irregular, and a filing that omitted only the
    suspicious ones would let the reconciliation rule stand in for the others.
    """
    rng = builder.rng
    lines: list[ReportedDonation] = []

    for donation in builder.donations:
        if builder.truth.get(donation.donation_id) == T_UNREPORTED:
            continue
        if rng.random() < config.unreported_share:
            # An ordinary omission. Detected as a reporting failure, which it
            # is, and not counted against the typology's own recall.
            continue
        lines.append(
            ReportedDonation(
                electoral_context=config.electoral_context,
                report_kind="LPSDK",
                donor_name=donation.sender_ref.raw_text,
                donor_ref=donation.sender_ref.entity_id,
                recipient_ref=donation.receiver_ref.entity_id,
                # Filed figures are transcribed by hand, so they carry the
                # rounding and slippage a real return would.
                amount_idr=int(donation.amount_idr * rng.uniform(0.995, 1.005)),
                occurred_on=donation.occurred_at.date()
                + timedelta(days=rng.randint(-1, 1)),
            )
        )

    return SubmissionSet(
        lines,
        covered_periods=(
            (config.electoral_context, config.period_start, config.period_end),
        ),
        available=True,
        # Fixture data standing in for filings held by the electoral authority.
        # A finding drawn from it demonstrates the rule; it establishes nothing.
        authoritative=False,
    )


def _convictions_register(builder: _Builder) -> Register:
    """A stand-in register of convictions with final legal force.

    Empty of real entries. The statute requires a conviction that is final, and
    no source available here can establish that for anyone, so the register is
    supplied but marked non-authoritative and carries only entities this
    generator itself created.
    """
    return Register(
        RegisterSet.FINAL_CONVICTIONS,
        (),
        available=True,
        authoritative=False,
        refreshed_at=datetime.now(tz=WIB),
    )


def _prohibited_register(builder: _Builder) -> Register:
    entries = [
        RegisterEntry(
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            category="state-enterprise"
            if entity.entity_type is EntityType.STATE_ENTERPRISE
            else "government",
            source="synthetic",
        )
        for entity in builder.entities.values()
        if "prohibited_source" in entity.registers
    ]
    return Register(
        RegisterSet.PROHIBITED_SOURCE,
        entries,
        available=True,
        # Generated data standing in for a register held by the electoral and
        # financial authorities. Findings drawn from it are demonstrations.
        authoritative=False,
        refreshed_at=datetime.now(tz=WIB),
    )
