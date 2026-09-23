import {
  FileText,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Plus,
  Settings,
  Trash2,
  Upload,
  X,
} from "lucide-react";

import type {
  ChatSession,
  DocumentRecord,
} from "../types/chat";

interface SidebarProps {
  open: boolean;
  collapsed: boolean;
  chats: ChatSession[];
  documents: DocumentRecord[];
  activeChatId: string | null;
  activeDocumentId: string | null;
  onNewChat: () => void;
  onSelectChat: (chat: ChatSession) => void;
  onDeleteChat: (chatId: string) => void;
  onRenameChat: (chatId: string) => void;
  onSelectDocument: (
    document: DocumentRecord,
  ) => void;
  onUploadClick: () => void;
  onSettings: () => void;
  onCloseMobile: () => void;
}

function formatRelativeDate(
  timestamp: number,
): string {
  const date = new Date(timestamp);
  const now = new Date();

  const diff =
    now.getTime() - date.getTime();

  const day =
    1000 * 60 * 60 * 24;

  if (diff < day) {
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  if (diff < day * 2) {
    return "Yesterday";
  }

  return date.toLocaleDateString([], {
    day: "2-digit",
    month: "short",
  });
}

export function Sidebar({
  open,
  collapsed,
  chats,
  documents,
  activeChatId,
  activeDocumentId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  onSelectDocument,
  onUploadClick,
  onSettings,
  onCloseMobile,
}: SidebarProps) {
  return (
    <>
      {open && (
        <button
          className="sidebar-mobile-backdrop"
          onClick={onCloseMobile}
          aria-label="Close sidebar"
        />
      )}

      <aside
        className={[
          "sidebar",
          open
            ? "sidebar-open"
            : "sidebar-closed",
          collapsed
            ? "sidebar-collapsed"
            : "",
        ].join(" ")}
      >
        <div className="sidebar-inner">
          <div className="sidebar-top">
            <div className="brand-row">
              {!collapsed && (
                <div className="brand">
                  <div className="brand-mark">
                    <span />
                    <span />
                    <span />
                  </div>

                  <span className="brand-name">
                    DocuRAG
                  </span>
                </div>
              )}

              <button
                className="icon-button mobile-only"
                onClick={onCloseMobile}
                aria-label="Close sidebar"
              >
                <X size={18} />
              </button>
            </div>

            <button
              className="new-chat-button"
              onClick={onNewChat}
            >
              <Plus size={18} />

              {!collapsed && (
                <span>New chat</span>
              )}
            </button>
          </div>

          {!collapsed && (
            <>
              <section className="sidebar-section">
                <div className="section-label">
                  Recent chats
                </div>

                <div className="chat-history">
                  {chats.length === 0 ? (
                    <div className="sidebar-empty">
                      <MessageSquare
                        size={18}
                      />
                      <span>
                        No conversations yet
                      </span>
                    </div>
                  ) : (
                    chats
                      .slice(0, 20)
                      .map((chat) => (
                        <div
                          key={chat.id}
                          className={[
                            "chat-history-item",
                            chat.id ===
                            activeChatId
                              ? "active"
                              : "",
                          ].join(" ")}
                        >
                          <button
                            className="chat-history-main"
                            onClick={() =>
                              onSelectChat(
                                chat,
                              )
                            }
                          >
                            <MessageSquare
                              size={16}
                            />

                            <div className="chat-history-copy">
                              <span className="chat-title">
                                {chat.title}
                              </span>

                              <span className="chat-time">
                                {formatRelativeDate(
                                  chat.updatedAt,
                                )}
                              </span>
                            </div>
                          </button>

                          <div className="chat-actions">
                            <button
                              className="mini-icon-button"
                              onClick={() =>
                                onRenameChat(
                                  chat.id,
                                )
                              }
                              aria-label="Rename chat"
                            >
                              <Pencil
                                size={13}
                              />
                            </button>

                            <button
                              className="mini-icon-button danger"
                              onClick={() =>
                                onDeleteChat(
                                  chat.id,
                                )
                              }
                              aria-label="Delete chat"
                            >
                              <Trash2
                                size={13}
                              />
                            </button>
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </section>

              <section className="sidebar-section documents-section">
                <div className="section-header">
                  <div className="section-label">
                    Documents
                  </div>

                  <button
                    className="mini-icon-button"
                    onClick={onUploadClick}
                    aria-label="Upload document"
                  >
                    <Plus size={14} />
                  </button>
                </div>

                <div className="document-list">
                  {documents.length === 0 ? (
                    <button
                      className="sidebar-upload-card"
                      onClick={onUploadClick}
                    >
                      <Upload size={17} />

                      <div>
                        <strong>
                          Upload a PDF
                        </strong>
                        <span>
                          Start analyzing a document
                        </span>
                      </div>
                    </button>
                  ) : (
                    documents
                      .slice(0, 10)
                      .map((document) => (
                        <button
                          key={document.id}
                          className={[
                            "document-sidebar-item",
                            document.id ===
                            activeDocumentId
                              ? "active"
                              : "",
                          ].join(" ")}
                          onClick={() =>
                            onSelectDocument(
                              document,
                            )
                          }
                        >
                          <div className="document-icon">
                            <FileText
                              size={16}
                            />
                          </div>

                          <div className="document-copy">
                            <span>
                              {document.name}
                            </span>

                            <small>
                              {formatFileSize(
                                document.size,
                              )}
                            </small>
                          </div>
                        </button>
                      ))
                  )}
                </div>
              </section>
            </>
          )}

          <div className="sidebar-bottom">
            <button
              className="sidebar-bottom-button"
              onClick={onSettings}
            >
              <Settings size={18} />

              {!collapsed && (
                <span>Settings</span>
              )}
            </button>

            <div className="profile">
              <div className="profile-avatar">
                VS
              </div>

              {!collapsed && (
                <div className="profile-copy">
                  <strong>
                    Vishwa Sabaris V
                  </strong>
                  <span>
                    Local workspace
                  </span>
                </div>
              )}

              {!collapsed && (
                <MoreHorizontal
                  size={17}
                  className="profile-more"
                />
              )}
            </div>
          </div>
        </div>
      </aside>
    </>
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
