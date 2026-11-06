from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from src.v3.market_data_readiness import (
    BAR_COLUMNS,
    CORPORATE_ACTION_COLUMNS,
    REQUIRED_ATTESTATIONS,
    REQUIRED_BINDINGS,
    SESSION_COLUMNS,
    STATUS_COLUMNS,
    SYMBOL_HISTORY_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "requests/setsmart_v3_20260901"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_schema_exactly_matches_the_live_intake_contract() -> None:
    schema = pd.read_csv(PACKET / "requested_five_file_schema.csv")
    expected = {
        "bars": BAR_COLUMNS,
        "trading_status": STATUS_COLUMNS,
        "sessions": SESSION_COLUMNS,
        "corporate_actions": CORPORATE_ACTION_COLUMNS,
        "symbol_history": SYMBOL_HISTORY_COLUMNS,
    }
    assert len(schema) == sum(len(columns) for columns in expected.values()) == 33
    assert not schema.duplicated(["component", "field"]).any()
    assert schema["required"].all()
    for component, columns in expected.items():
        observed = tuple(schema.loc[schema["component"].eq(component), "field"])
        assert observed == columns


def test_packet_union_is_an_exact_copy_of_the_canonical_138_tickers() -> None:
    packet_union = PACKET / "set50_set100_acquisition_union_138.csv"
    canonical_union = ROOT / "data/reference/set50_or_set100_universe.csv"
    assert packet_union.read_bytes() == canonical_union.read_bytes()
    frame = pd.read_csv(packet_union)
    assert len(frame) == 138
    assert frame["ticker"].nunique() == 138
    assert frame["usage"].eq("acquisition_only_not_static_model_membership").all()


def test_example_manifest_is_deliberately_ineligible_nondata() -> None:
    manifest = yaml.safe_load(
        (PACKET / "example_market_package_manifest_NONDATA.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["package_id"] == "EXAMPLE_NONDATA_DO_NOT_USE"
    assert manifest["license"] == {
        "reference": "EXAMPLE_ONLY_NO_LICENSE_GRANTED",
        "research_use_allowed": False,
        "derived_publication_allowed": False,
        "raw_redistribution_allowed": False,
    }
    assert tuple(manifest["bindings"]) == REQUIRED_BINDINGS
    assert tuple(manifest["attestations"]) == REQUIRED_ATTESTATIONS
    assert not any(manifest["attestations"].values())
    for binding in manifest["bindings"].values():
        assert binding["sha256"] == "0" * 64
        assert binding["bytes"] == 0
        assert binding["rows"] == 0


def test_packet_inventory_is_complete_and_self_bound() -> None:
    inventory = PACKET / "SHA256SUMS"
    sidecar = PACKET / "SHA256SUMS.sha256"
    sidecar_hash, sidecar_name = sidecar.read_text(encoding="utf-8").split()
    assert sidecar_name == inventory.name
    assert sidecar_hash == _sha256(inventory)

    entries: dict[str, str] = {}
    for line in inventory.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    assert tuple(entries) == tuple(sorted(entries))
    assert set(entries) == {
        "README.md",
        "cover_letter_DRAFT_UNSENT.md",
        "data_protection_license_checklist.md",
        "example_market_package_manifest_NONDATA.yaml",
        "requested_five_file_schema.csv",
        "set50_set100_acquisition_union_138.csv",
    }
    for name, digest in entries.items():
        assert digest == _sha256(PACKET / name)
    assert "DRAFT — UNSENT" in (PACKET / "README.md").read_text(encoding="utf-8")
    assert "DRAFT — UNSENT" in (
        PACKET / "cover_letter_DRAFT_UNSENT.md"
    ).read_text(encoding="utf-8")
