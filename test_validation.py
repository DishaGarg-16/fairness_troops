"""Test Pydantic validation in schemas."""
from api.schemas import DatasetConfig

print("=== Pydantic Validation Tests ===\n")

# Test 1: Valid configuration
print("1. Testing valid configuration...")
try:
    config = DatasetConfig(
        target_col="income",
        sensitive_col="gender",
        privileged_group="Male",
        unprivileged_group="Female"
    )
    print(f"   [PASS] Created: target={config.target_col}, sensitive={config.sensitive_col}")
except Exception as e:
    print(f"   [FAIL] Unexpected error: {e}")

# Test 2: Invalid column name (starts with number)
print("\n2. Testing invalid column name (starts with number)...")
try:
    config = DatasetConfig(
        target_col="123invalid",
        sensitive_col="gender",
        privileged_group="Male",
        unprivileged_group="Female"
    )
    print(f"   [FAIL] Should have rejected invalid column name")
except Exception as e:
    print(f"   [PASS] Correctly rejected: {type(e).__name__}")

# Test 3: Same target and sensitive column
print("\n3. Testing same target and sensitive column...")
try:
    config = DatasetConfig(
        target_col="income",
        sensitive_col="income",
        privileged_group="Male",
        unprivileged_group="Female"
    )
    print(f"   [FAIL] Should have rejected same columns")
except Exception as e:
    print(f"   [PASS] Correctly rejected: {type(e).__name__}")

# Test 4: Forbidden characters in group name
print("\n4. Testing forbidden characters in group name...")
try:
    config = DatasetConfig(
        target_col="income",
        sensitive_col="gender",
        privileged_group="Male<script>",
        unprivileged_group="Female"
    )
    print(f"   [FAIL] Should have rejected forbidden characters")
except Exception as e:
    print(f"   [PASS] Correctly rejected: {type(e).__name__}")

# Test 5: Empty column name
print("\n5. Testing empty column name...")
try:
    config = DatasetConfig(
        target_col="",
        sensitive_col="gender",
        privileged_group="Male",
        unprivileged_group="Female"
    )
    print(f"   [FAIL] Should have rejected empty column")
except Exception as e:
    print(f"   [PASS] Correctly rejected: {type(e).__name__}")

print("\n=== All tests completed ===")
