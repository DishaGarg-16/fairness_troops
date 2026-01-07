import pytest
from unittest.mock import patch, MagicMock
import io

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Fairness Troops API"}

def test_audit_validation_error(client):
    # Missing fields
    response = client.post("/audit", data={})
    assert response.status_code == 422

def test_audit_invalid_file_extension(client):
    # Valid form data
    data = {
        "target_col": "income",
        "sensitive_col": "gender",
        "privileged_group": "Male",
        "unprivileged_group": "Female"
    }
    # Invalid file extension
    files = {
        "model_file": ("model.pkl", b"fake model content", "application/octet-stream"),
        "data_file": ("data.csv", b"fake csv content", "text/csv")
    }
    
    response = client.post("/audit", data=data, files=files)
    assert response.status_code == 400
    assert "Invalid model file" in response.json()['detail']

@patch("api.main.run_audit_task.delay")
def test_audit_success(mock_task, client):
    # Valid data
    data = {
        "target_col": "income",
        "sensitive_col": "gender",
        "privileged_group": "Male",
        "unprivileged_group": "Female"
    }
    
    # Valid files
    # Create valid dummy content
    files = {
        "model_file": ("model.skops", b"fake skops content", "application/octet-stream"),
        "data_file": ("data.csv", b"header1,header2\nval1,val2", "text/csv")
    }

    # Mock task return
    mock_task_instance = MagicMock()
    mock_task_instance.id = "test-task-id"
    mock_task.return_value = mock_task_instance

    response = client.post("/audit", data=data, files=files)
    
    assert response.status_code == 200
    assert response.json()["task_id"] == "test-task-id"
    assert response.json()["status"] == "processing"
