import {
  Bot,
  Check,
  Copy,
  FileText,
  User,
} from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";

import type {
  ChatMessage,
  SourceReference,
} from "../types/chat";

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
  onExampleQuestion: (
    question: string,
  ) => void;
}

export function ChatWindow({
  messages,
  loading,
  onExampleQuestion,
}: ChatWindowProps) {
  const bottomRef =
    useRef<HTMLDivElement | null>(
      null,
    );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  if (messages.length === 0) {
    return (
      <div className="chat-empty-state">
        <div className="empty-logo">
          <div className="brand-mark large">
            <span />
            <span />
            <span />
          </div>
        </div>

        <h1>
          Analyze your documents
        </h1>

        <p>
          Ask questions, extract facts,
          compare information, and explore
          your PDF with grounded answers.
        </p>

        <div className="example-grid">
          <button
            onClick={() =>
              onExampleQuestion(
                "What is the main topic of this document?",
              )
            }
          >
            <FileText size={17} />
            <span>
              What is the main topic?
            </span>
          </button>

          <button
            onClick={() =>
              onExampleQuestion(
                "Summarize the key points of this document.",
              )
            }
          >
            <FileText size={17} />
            <span>
              Summarize the key points
            </span>
          </button>

          <button
            onClick={() =>
              onExampleQuestion(
                "What are the most important facts in this document?",
              )
            }
          >
            <FileText size={17} />
            <span>
              Extract important facts
            </span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="messages">
      {messages.map((message) => (
        <Message
          key={message.id}
          message={message}
        />
      ))}

      {loading && (
        <div className="message-row assistant-row">
          <div className="message-avatar assistant-avatar">
            <Bot size={17} />
          </div>

          <div className="message-body">
            <div className="message-author">
              DocuRAG
            </div>

            <div className="typing-indicator">
              <span />
              <span />
              <span />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

function Message({
  message,
}: {
  message: ChatMessage;
}) {
  const [copied, setCopied] =
    useState(false);

  async function copyAnswer() {
    await navigator.clipboard.writeText(
      message.content,
    );

    setCopied(true);

    window.setTimeout(
      () => setCopied(false),
      1500,
    );
  }

  const isUser =
    message.role === "user";

  return (
    <div
      className={[
        "message-row",
        isUser
          ? "user-row"
          : "assistant-row",
      ].join(" ")}
    >
      <div
        className={[
          "message-avatar",
          isUser
            ? "user-avatar"
            : "assistant-avatar",
        ].join(" ")}
      >
        {isUser ? (
          <User size={16} />
        ) : (
          <Bot size={17} />
        )}
      </div>

      <div className="message-body">
        <div className="message-author">
          {isUser
            ? "You"
            : "DocuRAG"}
        </div>

        <div
          className={[
            "message-content",
            message.isError
              ? "message-error"
              : "",
          ].join(" ")}
        >
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              components={{
                pre: ({
                  children,
                }) => (
                  <pre className="markdown-code">
                    {children}
                  </pre>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {!isUser &&
          message.sources &&
          message.sources.length > 0 && (
            <Sources
              sources={
                message.sources
              }
            />
          )}

        {!isUser && (
          <div className="message-actions">
            <button
              onClick={copyAnswer}
              title="Copy answer"
            >
              {copied ? (
                <Check size={14} />
              ) : (
                <Copy size={14} />
              )}

              <span>
                {copied
                  ? "Copied"
                  : "Copy"}
              </span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Sources({
  sources,
}: {
  sources: SourceReference[];
}) {
  return (
    <div className="sources">
      <div className="sources-title">
        <FileText size={14} />
        Sources
      </div>

      <div className="sources-list">
        {sources.map(
          (source, index) => (
            <div
              className="source-chip"
              key={
                source.chunk_id ||
                `${source.page_number}-${index}`
              }
            >
              <span>
                Source {index + 1}
              </span>

              {source.page_number !=
                null && (
                <small>
                  Page{" "}
                  {source.page_number}
                </small>
              )}
            </div>
          ),
        )}
      </div>
    </div>
  );
}
