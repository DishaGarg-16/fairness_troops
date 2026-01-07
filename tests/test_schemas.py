import pytest
from pydantic import ValidationError
from api.schemas import DatasetConfig

class TestDatasetConfig:
    def test_valid_configuration(self):
        """Test that a valid configuration is accepted."""
        config = DatasetConfig(
            target_col="income",
            sensitive_col="gender",
            privileged_group="Male",
            unprivileged_group="Female"
        )
        assert config.target_col == "income"
        assert config.sensitive_col == "gender"

    def test_invalid_column_name_starts_with_number(self):
        """Test that column names starting with a number are rejected."""
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                target_col="123invalid",
                sensitive_col="gender",
                privileged_group="Male",
                unprivileged_group="Female"
            )
        assert "must start with a letter" in str(excinfo.value)

    def test_same_target_and_sensitive_column(self):
        """Test that target and sensitive columns cannot be the same."""
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                target_col="income",
                sensitive_col="income",
                privileged_group="Male",
                unprivileged_group="Female"
            )
        assert "Target and sensitive columns cannot be the same" in str(excinfo.value)

    def test_forbidden_characters_in_group_name(self):
        """Test that group names with forbidden characters are rejected."""
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                target_col="income",
                sensitive_col="gender",
                privileged_group="Male<script>",
                unprivileged_group="Female"
            )
        assert "contains forbidden characters" in str(excinfo.value)

    def test_empty_column_name(self):
        """Test that empty column names are rejected."""
        with pytest.raises(ValidationError) as excinfo:
            DatasetConfig(
                target_col="",
                sensitive_col="gender",
                privileged_group="Male",
                unprivileged_group="Female"
            )
        assert "String should have at least 1 character" in str(excinfo.value)
