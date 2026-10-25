from copy import deepcopy
from pathlib import Path
from pydantic import ValidationError
from shared.modeling_plan import ModelingPlan

def expect_rejection(label: str, payload: dict)-> None:
  try:
    ModelingPlan.model_validate(payload)
  except ValidationError as error:
    message = error.errors()[0]["msg"]
    print(f"Rejected{label}: {message}")
    return

  raise AssertionError(
    f"Invalid plan was unexpectedly accepted: {label}"
  )

def main():
  plan_text= Path("configs/modeling_plan.json").read_text(encoding="utf-8")

  plan = ModelingPlan.model_validate_json(plan_text)

  print(f"Valid plan accepted: {len(plan.candidates)} candidates.")
  original = plan.model_dump()

  invalid_cases = [
    ("unsupported model", 1, "model_name", "xgboost"),
    ("wrong model parameter", 2, "learning_rate", 0.1),
    ("negative tree count", 2 , "n_estimators", -5),
    ("string tree count", 2,"n_estimators", "100"),
    ("duplicate candidate ID", 3,"candidate_id", "rf_depth4"),
  ]

  for label, index, field, value in invalid_cases:
    invalid_plan = deepcopy(original)
    invalid_plan["candidates"][index][field] = value
    expect_rejection(label, invalid_plan)

  missing_baseline = deepcopy(original)
  missing_baseline["candidates"] = (missing_baseline["candidates"][1:])

  expect_rejection("missing baseline", missing_baseline)

  print("\nModeling plan checks passed")

if __name__ == "__main__":
  main()