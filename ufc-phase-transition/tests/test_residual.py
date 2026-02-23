import pytest
import numpy as np
from src.residual import ResidualAnalyzer


def test_residual_analyzer_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ResidualAnalyzer([0.5, 0.6], [1])


def test_residual_analyzer_computes_residuals():
    analyzer = ResidualAnalyzer([0.7, 0.3], [1, 0])
    np.testing.assert_allclose(analyzer.residuals, [0.7 - 1, 0.3 - 0])


def test_residual_analyzer_summary_runs(capsys):
    analyzer = ResidualAnalyzer([0.7, 0.3, 0.6], [1, 0, 1])
    analyzer.summary()
    captured = capsys.readouterr()
    assert "Brier Score" in captured.out


def test_residual_group_by_requires_metadata():
    analyzer = ResidualAnalyzer([0.5], [1], metadata=None)
    with pytest.raises(ValueError, match="No metadata"):
        analyzer.group_by("weight_class")
