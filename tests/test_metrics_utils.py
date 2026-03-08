"""Tests for metrics_utils module."""

from biometric_recognition.utils.metrics_utils import (
    get_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)


class TestPlotTrainingHistory:
    """Tests for plot_training_history function."""

    def test_creates_plot_file(self, temp_dir):
        """Test that training history plot is created."""
        train_losses = [1.0, 0.8, 0.6, 0.4, 0.3]
        val_losses = [1.1, 0.9, 0.7, 0.5, 0.4]
        val_accuracies = [50.0, 60.0, 70.0, 80.0, 85.0]
        save_path = temp_dir / "training_history.png"

        plot_training_history(train_losses, val_losses, val_accuracies, save_path)

        assert save_path.exists()

    def test_handles_single_epoch(self, temp_dir):
        """Test that function handles single epoch data."""
        train_losses = [1.0]
        val_losses = [1.1]
        val_accuracies = [50.0]
        save_path = temp_dir / "training_history.png"

        plot_training_history(train_losses, val_losses, val_accuracies, save_path)

        assert save_path.exists()


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_creates_confusion_matrix_plot(self, temp_dir):
        """Test that confusion matrix plot is created."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]
        num_classes = 3
        save_path = temp_dir / "confusion_matrix.png"

        result_path = plot_confusion_matrix(y_true, y_pred, num_classes, save_path)

        assert save_path.exists()
        assert result_path == str(save_path)

    def test_handles_perfect_predictions(self, temp_dir):
        """Test with perfect predictions (diagonal confusion matrix)."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        num_classes = 3
        save_path = temp_dir / "confusion_matrix.png"

        plot_confusion_matrix(y_true, y_pred, num_classes, save_path)

        assert save_path.exists()

    def test_handles_many_classes(self, temp_dir):
        """Test with many classes (>20, which disables text annotations)."""
        num_classes = 25
        y_true = list(range(num_classes)) * 2
        y_pred = list(range(num_classes)) * 2
        save_path = temp_dir / "confusion_matrix.png"

        plot_confusion_matrix(y_true, y_pred, num_classes, save_path)

        assert save_path.exists()


class TestGetClassificationReport:
    """Tests for get_classification_report function."""

    def test_returns_dict_by_default(self):
        """Test that function returns dict by default."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]

        report = get_classification_report(y_true, y_pred)

        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "macro avg" in report
        assert "weighted avg" in report

    def test_returns_string_when_specified(self):
        """Test that function returns string when output_dict=False."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]

        report = get_classification_report(y_true, y_pred, output_dict=False)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report

    def test_handles_zero_division(self):
        """Test that function handles classes with no predictions."""
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]  # No correct predictions for class 0

        # Should not raise, due to zero_division=0
        report = get_classification_report(y_true, y_pred)

        assert isinstance(report, dict)
