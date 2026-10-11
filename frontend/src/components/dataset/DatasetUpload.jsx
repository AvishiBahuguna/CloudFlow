import { useRef, useState } from "react";
import { UploadCloud, FileSpreadsheet, X } from "lucide-react";
import { DATASET_CONFIG } from "../../constants/datasetConfig";

function DatasetUpload({ file, onFileSelect, error, disabled = false }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  function handleFile(selectedFile) {
    if (!selectedFile || disabled) return;
    onFileSelect(selectedFile);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const selectedFile = event.dataTransfer.files[0];
    handleFile(selectedFile);
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (!disabled) setIsDragging(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setIsDragging(false);
  }

  return (
    <section className="form-panel">
      <div className="form-panel-heading">
        <FileSpreadsheet size={20} />
        <div>
          <h2>Dataset Upload</h2>
          <p>Upload your machine learning dataset.</p>
        </div>
      </div>

      <div
        className={`upload-zone ${isDragging ? "dragging" : ""} ${error ? "has-error" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <UploadCloud size={36} />
        <p>Drag and drop your CSV file here</p>
        <span>or</span>

        <button type="button" className="secondary-button" disabled={disabled} onClick={() => inputRef.current?.click()}>
          Browse Files
        </button>

        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          disabled={disabled}
          className="visually-hidden"
          aria-label="Choose CSV dataset"
          onChange={(event) => {
            handleFile(event.target.files?.[0]);
            event.target.value = ""; // Allow selecting the same file again.
          }}
        />

        <small>CSV only · Maximum {DATASET_CONFIG.maxFileSizeMB} MB</small>
      </div>

      {file && (
        <div className="selected-file">
          <FileSpreadsheet size={20} />
          <div className="selected-file-info">
            <strong>{file.name}</strong>
            <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>

          <button type="button" className="remove-file" onClick={() => onFileSelect(null)} disabled={disabled} aria-label="Remove selected file">
            <X size={18} />
          </button>
        </div>
      )}

      {error && <p className="form-error" role="alert">{error}</p>}
    </section>
  );
}

export default DatasetUpload;
