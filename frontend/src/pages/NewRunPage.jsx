
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Papa from "papaparse";
import { ArrowRight, Loader2 } from "lucide-react";

import DatasetUpload from "../components/dataset/DatasetUpload";
import DatasetPreview from "../components/dataset/DatasetPreview";
import ProblemInput from "../components/dataset/ProblemInput";

import { DATASET_CONFIG } from "../constants/datasetConfig";
import { createPipelineRun } from "../services/pipelineService";

import "../components/dataset/DatasetForm.css";

function NewRunPage() {
  const navigate = useNavigate();

  // 1. Store the user's inputs
  const [file, setFile] = useState(null);
  const [objective, setObjective] = useState("");

  // 2. Store the CSV preview
  const [preview, setPreview] = useState(null);
  const [isParsing, setIsParsing] = useState(false);

  // 3. Track errors and submission
  const [fileError, setFileError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Ignore results from an older file selection
  const selectionVersion = useRef(0);

  function handleFileSelect(selectedFile) {
    const version = ++selectionVersion.current;

    // Clear the previous selection
    setFile(null);
    setPreview(null);
    setFileError("");
    setSubmitError("");
    setIsParsing(false);

    // The user clicked Remove
    if (!selectedFile) return;

    const maxBytes =
      DATASET_CONFIG.maxFileSizeMB * 1024 * 1024;

    // Check file extension
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFileError("Only CSV files are supported.");
      return;
    }

    // Check for empty file
    if (selectedFile.size === 0) {
      setFileError("The file is empty.");
      return;
    }

    // Check file size
    if (selectedFile.size > maxBytes) {
      setFileError(
        `Maximum file size is ${DATASET_CONFIG.maxFileSizeMB} MB.`
      );
      return;
    }

    setIsParsing(true);

    // Read the beginning of the CSV locally
    Papa.parse(selectedFile, {
      header: false,
      preview: DATASET_CONFIG.previewRows + 1,
      skipEmptyLines: "greedy",

      complete: (results) => {
        // Ignore outdated parsing results
        if (version !== selectionVersion.current) return;

        setIsParsing(false);

        if (results.errors.length > 0) {
          setFileError("Could not parse the CSV preview.");
          return;
        }

        const [columns = [], ...rows] = results.data;

        if (columns.length === 0 || rows.length === 0) {
          setFileError(
            "CSV must contain a header and at least one data row."
          );
          return;
        }

        if (columns.some((column) => !column.trim())) {
          setFileError("Column names cannot be empty.");
          return;
        }

        if (new Set(columns).size !== columns.length) {
          setFileError("Duplicate column names were detected.");
          return;
        }

        if (rows.some((row) => row.length !== columns.length)) {
          setFileError(
            "CSV preview contains rows with an incorrect number of values."
          );
          return;
        }

        // Convert each row into an object
        const formattedRows = rows.map((row) =>
          Object.fromEntries(
            columns.map((column, index) => [
              column,
              row[index],
            ])
          )
        );

        // Save the valid file and preview
        setFile(selectedFile);
        setPreview({
          columns,
          rows: formattedRows,
        });
      },

      error: () => {
        if (version !== selectionVersion.current) return;

        setIsParsing(false);
        setFileError("Unable to read the selected file.");
      },
    });
  }

  // Enable launch only when all inputs are valid
  const canLaunch =
    file !== null &&
    preview !== null &&
    objective.trim().length >= 10 &&
    !isParsing &&
    !isSubmitting;

  async function handleSubmit(event) {
    event.preventDefault();

    if (!canLaunch) return;

    setSubmitError("");
    setIsSubmitting(true);

    try {
      const result = await createPipelineRun({
        file,
        objective,
      });

      // Open the newly created pipeline
      navigate(
        `/pipeline/${encodeURIComponent(result.runId)}`
      );
    } catch (error) {
      setSubmitError(
        error.message || "Unable to start the pipeline."
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="new-run-page">
      <header className="new-run-header">
        <p className="page-label mono">
          CLOUDFLOW / NEW RUN
        </p>

        <h1>Start a new pipeline</h1>

        <p className="page-description">
          Upload your dataset and describe the
          machine learning problem you want to solve.
        </p>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="new-run-grid">
          <div className="dataset-column">
            <DatasetUpload
              file={file}
              onFileSelect={handleFileSelect}
              error={fileError}
              disabled={isSubmitting}
            />

            <DatasetPreview
              preview={preview}
              loading={isParsing}
            />
          </div>

          <ProblemInput
            value={objective}
            onChange={setObjective}
            disabled={isSubmitting}
          />
        </div>

        <div className="launch-footer">
          <p>
            Your dataset will be submitted for analysis
            when you launch the pipeline.
          </p>

          <button
            type="submit"
            className="launch-button"
            disabled={!canLaunch}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="spin-icon" size={18} />
                Starting...
              </>
            ) : (
              <>
                Launch Pipeline
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>

        {submitError && (
          <p className="form-error" role="alert">
            {submitError}
          </p>
        )}
      </form>
    </section>
  );
}

export default NewRunPage;
