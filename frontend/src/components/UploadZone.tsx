import {
  FileText,
  UploadCloud,
  X,
} from "lucide-react";
import {
  useRef,
  useState,
} from "react";

import type {
  DocumentRecord,
} from "../types/chat";

interface UploadZoneProps {
  documents: DocumentRecord[];
  uploading: boolean;
  onFiles: (
    files: File[],
  ) => void;
  onRemove: (
    documentId: string,
  ) => void;
}

export function UploadZone({
  documents,
  uploading,
  onFiles,
  onRemove,
}: UploadZoneProps) {
  const inputRef =
    useRef<HTMLInputElement | null>(
      null,
    );

  const [dragging, setDragging] =
    useState(false);

  function handleFiles(
    fileList: FileList | null,
  ) {
    if (!fileList) {
      return;
    }

    onFiles(
      Array.from(fileList),
    );
  }

  return (
    <div className="upload-zone-wrapper">
      <div
        className={[
          "upload-zone",
          dragging
            ? "dragging"
            : "",
        ].join(" ")}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() =>
          setDragging(false)
        }
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);

          handleFiles(
            event.dataTransfer.files,
          );
        }}
        onClick={() =>
          inputRef.current?.click()
        }
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(event) => {
            handleFiles(
              event.target.files,
            );

            event.target.value = "";
          }}
        />

        <div className="upload-icon">
          <UploadCloud size={28} />
        </div>

        <h2>
          Drop your PDF here
        </h2>

        <p>
          or click to browse from your
          computer
        </p>

        <span className="upload-hint">
          PDF documents only
        </span>

        {uploading && (
          <div className="upload-processing">
            Processing document...
          </div>
        )}
      </div>

      {documents.length > 0 && (
        <div className="uploaded-documents">
          {documents.map(
            (document) => (
              <div
                className="uploaded-document"
                key={document.id}
              >
                <div className="uploaded-document-icon">
                  <FileText
                    size={18}
                  />
                </div>

                <div className="uploaded-document-copy">
                  <strong>
                    {document.name}
                  </strong>

                  <span>
                    {formatFileSize(
                      document.size,
                    )}
                  </span>
                </div>

                <button
                  className="mini-icon-button"
                  onClick={(event) => {
                    event.stopPropagation();

                    onRemove(
                      document.id,
                    );
                  }}
                  aria-label="Remove document"
                >
                  <X size={15} />
                </button>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

function formatFileSize(
  bytes: number,
): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(
      bytes / 1024
    ).toFixed(1)} KB`;
  }

  return `${(
    bytes /
    (1024 * 1024)
  ).toFixed(1)} MB`;
}
