"""Node A — Extraction.

One parser per source. Each parser:

1. Reads a raw dict (as delivered by the vendor's JSON API).
2. Validates it against the vendor's known shape (strict, not best-effort).
3. Produces a list of ``Signal`` objects.
4. Rejected payloads are sent to the rejection log with a structured reason.

Parsers are pure-ish: no network I/O, no DB, no clock reads outside
``ingested_at`` defaults. HTTP clients live in a separate layer so
parsers can be unit-tested with fixture dicts.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable

from pydantic import ValidationError as PydanticValidationError

from .exceptions import ExtractionError, SchemaValidationError
from .logging_setup import audit, reject
from .models import Confidence, Signal, SignalKind, SignalSource
from .time_utils import EST


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _to_est(iso_string: str) -> datetime:
    """Parse an ISO-8601 timestamp and convert to EST.

    Vendors deliver in assorted zones; we normalize once here so the rest
    of the engine only sees EST datetimes.
    """
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(
            "unparseable timestamp", raw=iso_string
        ) from exc
    if dt.tzinfo is None:
        raise SchemaValidationError(
            "timestamp missing timezone", raw=iso_string
        )
    return dt.astimezone(EST)


def _to_decimal(v: object, field: str) -> Decimal:
    try:
        return Decimal(str(v))
    except (InvalidOperation, TypeError) as exc:
        raise SchemaValidationError(
            "value is not a decimal", field=field, raw=v
        ) from exc


def _pct_to_bps(pct: Decimal) -> int:
    """Convert a percentage (e.g. 5.25) to integer basis points (525).

    We multiply in Decimal space and then quantize — no float rounding.
    Fractional bps are rejected: the engine's contract is integer bps.
    """
    bps = pct * Decimal("100")
    if bps != bps.to_integral_value():
        raise SchemaValidationError(
            "rate has sub-bps precision — engine requires integer bps",
            pct=str(pct),
        )
    return int(bps)


def _build_signal(**kwargs) -> Signal:
    """Wrap pydantic errors into our SchemaValidationError hierarchy."""
    try:
        return Signal(**kwargs)
    except PydanticValidationError as exc:
        raise SchemaValidationError(
            "signal schema validation failed",
            errors=exc.errors(),
            kwargs=kwargs,
        ) from exc


def _safe_parse(
    source: SignalSource, payload: dict, parser
) -> list[Signal]:
    """Common envelope: catch and log, never raise from here."""
    try:
        signals = list(parser(payload))
        audit(
            "node_a.parsed",
            source=source.value,
            n_signals=len(signals),
        )
        return signals
    except (SchemaValidationError, ExtractionError) as exc:
        reject(
            "node_a.rejected",
            source=source.value,
            reason=exc.message,
            context=exc.context,
        )
        return []


# --------------------------------------------------------------------------- #
# FRED                                                                        #
# --------------------------------------------------------------------------- #
class FREDParser:
    """Federal Reserve Economic Data — rates, yields, real yields.

    Expected payload shape (trimmed):
      {"series_id": "DFF", "unit": "pct",
       "observations": [{"date": "2026-04-17", "value": "5.33"}, ...]}
    """

    _SERIES: dict[str, tuple[SignalKind, str]] = {
        "DFF":   (SignalKind.POLICY_RATE, "USD_EFFR"),
        "DGS10": (SignalKind.TREASURY_YIELD, "UST_10Y"),
        "DGS2":  (SignalKind.TREASURY_YIELD, "UST_2Y"),
        "DFII10":(SignalKind.REAL_YIELD, "UST_TIPS_10Y"),
    }

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.FRED, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        series_id = payload.get("series_id")
        if series_id not in self._SERIES:
            raise ExtractionError("unknown FRED series", series_id=series_id)
        kind, entity = self._SERIES[series_id]

        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            raise SchemaValidationError("observations missing or empty")

        for obs in observations:
            if obs.get("value") in (None, "", "."):
                # FRED uses "." for missing. Skip rather than reject the batch.
                continue
            pct = _to_decimal(obs["value"], field="observations[].value")
            yield _build_signal(
                source=SignalSource.FRED,
                kind=kind,
                entity=entity,
                value_bps=_pct_to_bps(pct),
                unit="bps",
                observed_at=_to_est(obs["date"] + "T00:00:00+00:00")
                if "T" not in obs["date"] else _to_est(obs["date"]),
                confidence=Confidence.HIGH,
                raw_payload=obs,
            )


# --------------------------------------------------------------------------- #
# Tiingo                                                                      #
# --------------------------------------------------------------------------- #
class TiingoParser:
    """Tiingo FX — spot, bid/ask, volume.

    Expected:
      {"ticker": "eurusd", "quote": {
         "timestamp": "2026-04-17T12:34:56Z",
         "bid": "1.0832", "ask": "1.0834", "mid": "1.0833",
         "volume": 2345000}}
    """

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.TIINGO, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        ticker = payload.get("ticker")
        quote = payload.get("quote")
        if not ticker or not isinstance(quote, dict):
            raise SchemaValidationError("missing ticker or quote")
        entity = ticker.upper()
        if "/" not in entity:
            entity = f"{entity[:3]}/{entity[3:]}"
        observed = _to_est(quote["timestamp"])

        mid = _to_decimal(quote["mid"], field="quote.mid")
        bid = _to_decimal(quote["bid"], field="quote.bid")
        ask = _to_decimal(quote["ask"], field="quote.ask")
        if bid > ask:
            raise SchemaValidationError(
                "inverted spread", bid=str(bid), ask=str(ask)
            )
        spread_bps = int(((ask - bid) / mid) * Decimal("10000"))

        yield _build_signal(
            source=SignalSource.TIINGO, kind=SignalKind.FX_SPOT,
            entity=entity, value_decimal=mid, unit="price",
            observed_at=observed, raw_payload=quote,
        )
        yield _build_signal(
            source=SignalSource.TIINGO, kind=SignalKind.FX_SPREAD,
            entity=entity, value_bps=spread_bps, unit="bps",
            observed_at=observed, raw_payload=quote,
        )
        if "volume" in quote and quote["volume"] is not None:
            yield _build_signal(
                source=SignalSource.TIINGO, kind=SignalKind.FX_VOLUME,
                entity=entity,
                value_decimal=_to_decimal(quote["volume"], "quote.volume"),
                unit="notional", observed_at=observed, raw_payload=quote,
            )


# --------------------------------------------------------------------------- #
# CME                                                                         #
# --------------------------------------------------------------------------- #
class CMEParser:
    """CME Fed Funds futures → implied rate path (basis points).

    Expected:
      {"contract": "ZQZ6", "observed_at": "...",
       "settlement_price": "94.75", "expiry": "2026-12-31"}
    Implied rate = (100 - price), expressed in bps.
    """

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.CME, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        price = _to_decimal(payload["settlement_price"], "settlement_price")
        if price <= 0 or price >= Decimal("100"):
            raise SchemaValidationError(
                "settlement price out of range", price=str(price)
            )
        implied_pct = Decimal("100") - price
        yield _build_signal(
            source=SignalSource.CME, kind=SignalKind.IMPLIED_RATE_PATH,
            entity=f"FFF_{payload['contract']}",
            value_bps=_pct_to_bps(implied_pct), unit="bps",
            observed_at=_to_est(payload["observed_at"]),
            raw_payload=payload,
        )


# --------------------------------------------------------------------------- #
# CFTC                                                                        #
# --------------------------------------------------------------------------- #
class CFTCParser:
    """Commitments of Traders — net positioning by category.

    Expected:
      {"market": "EURO FX", "report_date": "2026-04-15",
       "noncomm_long": 180000, "noncomm_short": 120000, "open_interest": 800000}
    Emits net positioning as a Decimal ratio in [-1, 1].
    """

    _MARKET: dict[str, str] = {
        "EURO FX": "EUR/USD",
        "JAPANESE YEN": "USD/JPY",
        "BRITISH POUND": "GBP/USD",
    }

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.CFTC, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        market = payload.get("market")
        if market not in self._MARKET:
            raise ExtractionError("unknown CFTC market", market=market)
        oi = _to_decimal(payload["open_interest"], "open_interest")
        if oi <= 0:
            raise SchemaValidationError("non-positive open interest")
        net = (
            _to_decimal(payload["noncomm_long"], "noncomm_long")
            - _to_decimal(payload["noncomm_short"], "noncomm_short")
        )
        ratio = (net / oi).quantize(Decimal("0.0001"))
        if ratio < Decimal("-1") or ratio > Decimal("1"):
            raise SchemaValidationError("net/OI ratio out of range")
        yield _build_signal(
            source=SignalSource.CFTC, kind=SignalKind.COT_POSITIONING,
            entity=self._MARKET[market], value_decimal=ratio,
            unit="ratio", observed_at=_to_est(payload["report_date"] + "T16:30:00-04:00"),
            raw_payload=payload,
        )


# --------------------------------------------------------------------------- #
# ECB / BoJ                                                                   #
# --------------------------------------------------------------------------- #
class CentralBankParser:
    """ECB / BoJ policy rate + forward guidance blobs."""

    def __init__(self, source: SignalSource, entity: str) -> None:
        if source not in {SignalSource.ECB, SignalSource.BOJ}:
            raise ExtractionError(
                "CentralBankParser only supports ECB/BoJ",
                source=source.value,
            )
        self._source = source
        self._entity = entity

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(self._source, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        observed = _to_est(payload["observed_at"])
        if "policy_rate_pct" in payload:
            pct = _to_decimal(payload["policy_rate_pct"], "policy_rate_pct")
            yield _build_signal(
                source=self._source, kind=SignalKind.POLICY_RATE,
                entity=self._entity, value_bps=_pct_to_bps(pct), unit="bps",
                observed_at=observed, raw_payload=payload,
            )
        guidance = payload.get("forward_guidance")
        if guidance:
            if not isinstance(guidance, dict) or "tone_score" not in guidance:
                raise SchemaValidationError(
                    "forward_guidance must include tone_score",
                    raw=guidance,
                )
            tone = _to_decimal(guidance["tone_score"], "tone_score")
            if tone < Decimal("-1") or tone > Decimal("1"):
                raise SchemaValidationError("tone_score out of [-1, 1]")
            yield _build_signal(
                source=self._source, kind=SignalKind.FORWARD_GUIDANCE,
                entity=self._entity, value_decimal=tone, unit="tone",
                observed_at=observed, confidence=Confidence.MEDIUM,
                raw_payload=payload,
            )


# --------------------------------------------------------------------------- #
# CBOE                                                                        #
# --------------------------------------------------------------------------- #
class CBOEParser:
    """VIX / MOVE / FX option IV."""

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.CBOE, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        index = payload.get("index")
        if index not in {"VIX", "MOVE", "EVZ"}:
            raise ExtractionError("unknown CBOE index", index=index)
        yield _build_signal(
            source=SignalSource.CBOE, kind=SignalKind.VOLATILITY,
            entity=index,
            value_decimal=_to_decimal(payload["level"], "level"),
            unit="index",
            observed_at=_to_est(payload["observed_at"]),
            raw_payload=payload,
        )


# --------------------------------------------------------------------------- #
# Economic calendar                                                           #
# --------------------------------------------------------------------------- #
class CalendarParser:
    """NFP / CPI / PMI / unemployment / retail sales releases.

    Emits the *surprise* (actual - consensus) in whatever unit the series
    ships — kept as Decimal, never converted to bps.
    """

    _SUPPORTED = {"NFP", "CPI_YOY", "PMI", "UNEMPLOYMENT", "RETAIL_SALES"}

    def parse(self, payload: dict) -> list[Signal]:
        return _safe_parse(SignalSource.CALENDAR, payload, self._parse_strict)

    def _parse_strict(self, payload: dict) -> Iterable[Signal]:
        series = payload.get("series")
        if series not in self._SUPPORTED:
            raise ExtractionError("unsupported calendar series", series=series)
        actual = _to_decimal(payload["actual"], "actual")
        consensus = _to_decimal(payload["consensus"], "consensus")
        surprise = actual - consensus
        yield _build_signal(
            source=SignalSource.CALENDAR, kind=SignalKind.ECON_RELEASE,
            entity=series, value_decimal=surprise, unit="surprise",
            observed_at=_to_est(payload["release_time"]),
            raw_payload=payload,
        )


# --------------------------------------------------------------------------- #
# Facade                                                                      #
# --------------------------------------------------------------------------- #
class Extractor:
    """Single entry point for orchestration."""

    def __init__(self) -> None:
        self._fred = FREDParser()
        self._tiingo = TiingoParser()
        self._cme = CMEParser()
        self._cftc = CFTCParser()
        self._ecb = CentralBankParser(SignalSource.ECB, "EUR_POLICY")
        self._boj = CentralBankParser(SignalSource.BOJ, "JPY_POLICY")
        self._cboe = CBOEParser()
        self._calendar = CalendarParser()

    def extract(self, source: SignalSource, payload: dict) -> list[Signal]:
        parser = {
            SignalSource.FRED: self._fred.parse,
            SignalSource.TIINGO: self._tiingo.parse,
            SignalSource.CME: self._cme.parse,
            SignalSource.CFTC: self._cftc.parse,
            SignalSource.ECB: self._ecb.parse,
            SignalSource.BOJ: self._boj.parse,
            SignalSource.CBOE: self._cboe.parse,
            SignalSource.CALENDAR: self._calendar.parse,
        }.get(source)
        if parser is None:
            reject(
                "node_a.unknown_source",
                source=source.value,
                reason="no parser registered",
            )
            return []
        return parser(payload)
