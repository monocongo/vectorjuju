"""The label-fidelity mark rule: marks may be lost, never moved or invented."""

from __future__ import annotations

import pytest
from acceptance_helpers import label_passes

TRUTH = "N 12°34'56\" E"


def test_an_exact_read_passes():
    assert label_passes(TRUTH, TRUTH)


@pytest.mark.parametrize(
    "recovered",
    [
        'N 123456" E',  # lost the degree and minute marks
        "N 12°34'56 E",  # lost the second mark
        "N 12°34'56” E",  # unicode double prime folds to the same mark
    ],
)
def test_lost_marks_still_pass(recovered):
    assert label_passes(recovered, TRUTH)


@pytest.mark.parametrize(
    "recovered",
    [
        "N 12'34°56\" E",  # degree and minute exchanged
        "N 12°34\"56' E",  # minute and second exchanged
        "N 12\"34'56° E",  # degree and second exchanged
        "N 1234°'56\" E",  # marks slide across the numbers, order kept
        "N 12°3456' E",  # near miss: minute moved after the seconds
        "N 12°34'56'' E",  # near miss: duplicated second mark
        "N 13°34'56\" E",  # near miss: wrong digit, marks untouched
    ],
)
def test_moved_or_invented_marks_fail(recovered):
    assert not label_passes(recovered, TRUTH)
