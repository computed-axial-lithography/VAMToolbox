"""Options: per-method defaults are per-instance, not rewritten by earlier kwargs."""
import copy

from vamtoolbox.optimize import Options


def _class_defaults():
    """Snapshot of the class-level default dicts (name-mangled attributes)."""
    return {
        name: copy.deepcopy(getattr(Options, "_Options__default_%s" % name))
        for name in ("FBP", "CAL", "PM", "OSMO", "BCLP")
    }


def test_kwargs_do_not_change_later_defaults():
    Options(method="BCLP", learning_rate=0.5, verbose="off")
    later = Options(method="BCLP", verbose="off")
    assert later.learning_rate == 0.01


def test_kwargs_do_not_leak_across_methods():
    Options(method="BCLP", learning_rate=0.5, verbose="off")
    cal = Options(method="CAL", verbose="off")
    assert cal.learning_rate == 0.01


def test_class_defaults_are_not_mutated():
    before = _class_defaults()
    Options(method="BCLP", learning_rate=0.5, eps=0.9, weight=7, verbose="off")
    Options(method="CAL", momentum=0.3, sigmoid=0.5, verbose="off")
    Options(method="PM", rho_1=9, verbose="off")
    Options(method="OSMO", inhibition=0.4, verbose="off")
    Options(method="FBP", offset=True)
    assert _class_defaults() == before


def test_kwargs_still_apply_to_their_own_instance():
    """Backward compatibility: the instance given the kwarg still receives it."""
    bclp = Options(method="BCLP", learning_rate=0.5, eps=0.9, weight=7, q=3, verbose="off")
    assert (bclp.learning_rate, bclp.eps, bclp.weight, bclp.q) == (0.5, 0.9, 7, 3)

    cal = Options(method="CAL", learning_rate=0.2, momentum=0.3, positivity=0.4, sigmoid=0.5)
    assert (cal.learning_rate, cal.momentum, cal.positivity, cal.sigmoid) == (0.2, 0.3, 0.4, 0.5)

    pm = Options(method="PM", rho_1=9, rho_2=8, p=4)
    assert (pm.rho_1, pm.rho_2, pm.p) == (9, 8, 4)

    osmo = Options(method="OSMO", inhibition=0.4)
    assert osmo.inhibition == 0.4

    fbp = Options(method="FBP", offset=True)
    assert fbp.offset is True


def test_unknown_kwargs_are_still_stored_on_the_instance():
    opt = Options(method="CAL", some_extra_setting=123)
    assert opt.some_extra_setting == 123


def test_explicit_response_model_does_not_become_the_default():
    from vamtoolbox import response

    model = response.ResponseModel()
    Options(method="BCLP", response_model=model, verbose="off")
    later = Options(method="BCLP", verbose="off")
    assert later.response_model is not model
