import pytest

from zetaline.risk.limits import RiskLimits


def test_defaults():
    lim = RiskLimits()
    assert lim.max_order_qty == 1_000.0
    assert lim.max_order_notional == 10_000.0
    assert lim.max_position_notional == 25_000.0
    assert lim.max_gross_exposure == 50_000.0
    assert lim.max_loss == 2_000.0
    assert lim.stale_tick_seconds == 2.0


def test_to_dict_has_all_fields():
    d = RiskLimits().to_dict()
    assert set(d) == {
        "max_order_qty",
        "max_order_notional",
        "max_position_notional",
        "max_gross_exposure",
        "max_loss",
        "stale_tick_seconds",
    }


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_non_positive_or_non_finite(bad):
    with pytest.raises(ValueError):
        RiskLimits(max_loss=bad)


@pytest.mark.parametrize("bad", ["abc", None, True, [1]])
def test_constructor_rejects_wrong_type(bad):
    with pytest.raises(TypeError):
        RiskLimits(max_loss=bad)


def test_ordering_constraint_enforced():
    with pytest.raises(ValueError):
        RiskLimits(max_order_notional=30_000.0)  # > max_position_notional
    with pytest.raises(ValueError):
        RiskLimits(max_position_notional=60_000.0)  # > max_gross_exposure


def test_update_returns_old_and_new():
    lim = RiskLimits()
    applied = lim.update(max_loss=1_000.0)
    assert applied == {"max_loss": (2_000.0, 1_000.0)}
    assert lim.max_loss == 1_000.0


def test_update_skips_unchanged_values():
    lim = RiskLimits()
    assert lim.update(max_loss=2_000.0) == {}


def test_update_multiple_fields():
    lim = RiskLimits()
    applied = lim.update(max_loss=500.0, stale_tick_seconds=5.0)
    assert set(applied) == {"max_loss", "stale_tick_seconds"}
    assert lim.stale_tick_seconds == 5.0


def test_update_unknown_field_rejected():
    lim = RiskLimits()
    before = lim.to_dict()
    with pytest.raises(ValueError, match="unknown"):
        lim.update(bogus=1)
    assert lim.to_dict() == before


def test_update_is_atomic_on_value_error():
    lim = RiskLimits()
    before = lim.to_dict()
    # First change is valid, second is not: nothing may be applied.
    with pytest.raises(ValueError):
        lim.update(max_loss=500.0, max_order_notional=99_999.0)
    assert lim.to_dict() == before


def test_update_is_atomic_on_type_error():
    lim = RiskLimits()
    before = lim.to_dict()
    with pytest.raises(TypeError):
        lim.update(max_loss=500.0, stale_tick_seconds="soon")
    assert lim.to_dict() == before


def test_update_respects_ordering_constraint():
    lim = RiskLimits()
    before = lim.to_dict()
    with pytest.raises(ValueError):
        lim.update(max_gross_exposure=20_000.0)  # < max_position_notional
    assert lim.to_dict() == before
    # Raising the ceiling together with the dependent limit is fine.
    lim.update(max_position_notional=15_000.0, max_gross_exposure=20_000.0)
    assert lim.max_gross_exposure == 20_000.0
