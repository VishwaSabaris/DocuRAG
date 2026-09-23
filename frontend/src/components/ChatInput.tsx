import {
  ArrowUp,
  FilePlus2,
  Loader2,
  Mic,
  Paperclip,
} from "lucide-react";

import type {
  DocumentRecord,
} from "../types/chat";

interface ChatInputProps {
  value: string;
  disabled?: boolean;
  loading?: boolean;
  activeDocument: DocumentRecord | null;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onAttach: () => void;
}

export function ChatInput({
  value,
  disabled = false,
  loading = false,
  activeDocument,
  onChange,
  onSubmit,
  onAttach,
}: ChatInputProps) {
  const canSend =
    value.trim().length > 0 &&
    !disabled &&
    !loading;

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      if (canSend) {
        onSubmit();
      }
    }
  }

  function handleInput(
    event: React.FormEvent<HTMLTextAreaElement>,
  ) {
    const textarea =
      event.currentTarget;

    textarea.style.height = "auto";

    textarea.style.height =
      `${Math.min(
        textarea.scrollHeight,
        180,
      )}px`;

    onChange(textarea.value);
  }

  return (
    <div className="composer-wrapper">
      <div className="composer">
        {activeDocument && (
          <div className="composer-document">
            <div className="composer-document-icon">
              <FilePlus2 size={14} />
            </div>

            <span>
              {activeDocument.name}
            </span>

            <span className="composer-document-status">
              Ready
            </span>
          </div>
        )}

        <div className="composer-main">
          <button
            className="composer-icon-button"
            onClick={onAttach}
            disabled={disabled}
            aria-label="Attach document"
          >
            <Paperclip size={19} />
          </button>

          <textarea
            value={value}
            rows={1}
            placeholder={
              activeDocument
                ? "Ask anything about this document..."
                : "Upload a document to start..."
            }
            disabled={disabled}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
          />

          <button
            className="composer-icon-button voice-button"
            disabled
            aria-label="Voice input"
            title="Voice input coming soon"
          >
            <Mic size={18} />
          </button>

          <button
            className={[
              "send-button",
              canSend
                ? "send-enabled"
                : "",
            ].join(" ")}
            onClick={onSubmit}
            disabled={!canSend}
            aria-label="Send message"
          >
            {loading ? (
              <Loader2
                size={18}
                className="spin"
              />
            ) : (
              <ArrowUp size={19} />
            )}
          </button>
        </div>

        <div className="composer-footer">
          <span>
            DocuRAG answers using your
            document context.
          </span>

          <span className="composer-shortcut">
            Enter to send · Shift + Enter
            for newline
          </span>
        </div>
      </div>
    </div>
  );
}
