
import { BrainCircuit } from "lucide-react";

import {
  OBJECTIVE_SUGGESTIONS,
} from "../../constants/datasetConfig";

function ProblemInput({
  value,
  onChange,
  disabled = false,
}) {
  return (
    <section className="form-panel">
      <div className="form-panel-heading">
        <BrainCircuit size={20} />

        <div>
          <h2>ML Objective</h2>
          <p>Describe what you want the agents to do.</p>
        </div>
      </div>

      <label
        htmlFor="ml-objective"
        className="field-label"
      >
        Problem statement
      </label>

      <textarea
        id="ml-objective"
        className="objective-input"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Example: Predict customer churn using the features in the uploaded dataset..."
        rows={6}
        maxLength={2000}
        required
      />

      <div className="input-hint">
        {value.length} / 2000 characters
      </div>

      <p className="suggestion-heading">
        Try an example
      </p>

      <div className="suggestion-list">
        {OBJECTIVE_SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.label}
            type="button"
            className="suggestion-button"
            disabled={disabled}
            onClick={() => onChange(suggestion.text)}
          >
            {suggestion.label}
          </button>
        ))}
      </div>
    </section>
  );
}

export default ProblemInput;
