
import { Table2 } from "lucide-react";

function DatasetPreview({ preview, loading }) {
  if (loading) {
    return (
      <section className="form-panel" aria-live="polite">
        <p>Reading dataset preview...</p>
      </section>
    );
  }

  if (!preview) {
    return null;
  }

  const { columns, rows } = preview;

  return (
    <section className="form-panel">
      <div className="form-panel-heading">
        <Table2 size={20} />

        <div>
          <h2>Dataset Preview</h2>
          <p>
            Showing up to {rows.length} sample rows.
          </p>
        </div>
      </div>

      <div className="dataset-summary">
        <span>
          <strong>{columns.length}</strong> columns
        </span>

        <span>
          <strong>{rows.length}</strong> preview rows
        </span>
      </div>

      <div className="table-scroll">
        <table className="preview-table">
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th key={`${column}-${index}`} scope="col">
                  {column}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column, colIndex) => (
                  <td key={`${column}-${colIndex}`}>
                    {row[column] === "" ||
                    row[column] == null
                      ? "—"
                      : String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="preview-note">
        This is a local preview. Full dataset profiling
        will be performed by the backend.
      </p>
    </section>
  );
}

export default DatasetPreview;
