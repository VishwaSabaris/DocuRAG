import {
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
} from "lucide-react";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import "./App.css";

import {
  ChatInput,
} from "./components/ChatInput";

import {
  ChatWindow,
} from "./components/ChatWindow";

import {
  SettingsModal,
} from "./components/SettingsModal";

import {
  Sidebar,
} from "./components/Sidebar";

import {
  UploadZone,
} from "./components/UploadZone";

import {
  askDocument,
  uploadPDF,
} from "./services/api";

import {
  createChat,
  createId,
  deleteChat,
  loadChats,
  loadDocuments,
  loadSettings,
  removeDocument,
  renameChat,
  saveSettings,
  updateChat,
  upsertDocument,
} from "./services/chatStorage";

import type {
  AppSettings,
  ChatMessage,
  ChatSession,
  DocumentRecord,
} from "./types/chat";

function App() {
  const [settings, setSettings] =
    useState<AppSettings>(
      loadSettings(),
    );

  const [chats, setChats] =
    useState<ChatSession[]>(
      loadChats(),
    );

  const [documents, setDocuments] =
    useState<DocumentRecord[]>(
      loadDocuments(),
    );

  const [activeChatId, setActiveChatId] =
    useState<string | null>(
      null,
    );

  const [
    activeDocumentId,
    setActiveDocumentId,
  ] = useState<string | null>(
    null,
  );

  const [input, setInput] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  const [uploading, setUploading] =
    useState(false);

  const [
    settingsOpen,
    setSettingsOpen,
  ] = useState(false);

  const [
    sidebarOpen,
    setSidebarOpen,
  ] = useState(true);

  const [
    sidebarCollapsed,
    setSidebarCollapsed,
  ] = useState(false);

  const fileInputRef =
    useRef<HTMLInputElement | null>(
      null,
    );

  const activeChat =
    chats.find(
      (chat) =>
        chat.id === activeChatId,
    ) || null;

  const activeDocument =
    documents.find(
      (document) =>
        document.id ===
        activeDocumentId,
    ) || null;

  const currentMessages =
    activeChat?.messages || [];

  const hasDocument =
    Boolean(activeDocument);

  useEffect(() => {
    saveSettings(settings);

    document.documentElement.dataset.theme =
      settings.theme;
  }, [settings]);

  useEffect(() => {
    const handleResize = () => {
      if (
        window.innerWidth < 900
      ) {
        setSidebarOpen(false);
        setSidebarCollapsed(false);
      } else {
        setSidebarOpen(true);
      }
    };

    handleResize();

    window.addEventListener(
      "resize",
      handleResize,
    );

    return () =>
      window.removeEventListener(
        "resize",
        handleResize,
      );
  }, []);

  function refreshChats() {
    setChats(loadChats());
  }

  function refreshDocuments() {
    setDocuments(loadDocuments());
  }

  function handleNewChat() {
    const chat = createChat(
      activeDocumentId
        ? [activeDocumentId]
        : [],
    );

    updateChat(chat);

    setChats([
      chat,
      ...chats,
    ]);

    setActiveChatId(chat.id);

    setInput("");

    if (
      window.innerWidth < 900
    ) {
      setSidebarOpen(false);
    }
  }

  function ensureActiveChat(): ChatSession {
    if (activeChat) {
      return activeChat;
    }

    const chat = createChat(
      activeDocumentId
        ? [activeDocumentId]
        : [],
    );

    updateChat(chat);

    setChats([
      chat,
      ...chats,
    ]);

    setActiveChatId(chat.id);

    return chat;
  }

  async function handleSend(
    forcedQuestion?: string,
  ) {
    const question = (
      forcedQuestion ??
      input
    ).trim();

    if (!question || loading) {
      return;
    }

    if (!activeDocument) {
      showToast(
        "Upload a document before asking a question.",
      );

      return;
    }

    const chat =
      ensureActiveChat();

    const userMessage: ChatMessage =
      {
        id: createId("message"),
        role: "user",
        content: question,
        createdAt: Date.now(),
      };

    const updatedMessages = [
      ...chat.messages,
      userMessage,
    ];

    const title =
      chat.messages.length === 0
        ? createChatTitle(question)
        : chat.title;

    const updatedChat: ChatSession =
      {
        ...chat,
        title,
        updatedAt: Date.now(),
        documentIds:
          chat.documentIds.includes(
            activeDocument.id,
          )
            ? chat.documentIds
            : [
                ...chat.documentIds,
                activeDocument.id,
              ],
        messages: updatedMessages,
      };

    updateChat(updatedChat);

    setChats((previous) =>
      previous
        .map((item) =>
          item.id === updatedChat.id
            ? updatedChat
            : item,
        )
        .sort(
          (a, b) =>
            b.updatedAt -
            a.updatedAt,
        ),
    );

    setInput("");
    setLoading(true);

    try {
      const response =
        await askDocument(
          activeDocument.id,
          question,
          settings.apiBaseUrl,
        );

      const assistantMessage:
        ChatMessage = {
          id: createId("message"),
          role: "assistant",
          content:
            response.answer ||
            "The information is not available in the provided document.",
          createdAt: Date.now(),
          sources:
            response.sources || [],
        };

      const finalChat:
        ChatSession = {
          ...updatedChat,
          updatedAt: Date.now(),
          messages: [
            ...updatedMessages,
            assistantMessage,
          ],
        };

      updateChat(finalChat);

      setChats((previous) =>
        previous
          .map((item) =>
            item.id === finalChat.id
              ? finalChat
              : item,
          )
          .sort(
            (a, b) =>
              b.updatedAt -
              a.updatedAt,
          ),
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Something went wrong.";

      const errorMessage:
        ChatMessage = {
          id: createId("message"),
          role: "assistant",
          content: message,
          createdAt: Date.now(),
          isError: true,
        };

      const failedChat:
        ChatSession = {
          ...updatedChat,
          updatedAt: Date.now(),
          messages: [
            ...updatedMessages,
            errorMessage,
          ],
        };

      updateChat(failedChat);

      setChats((previous) =>
        previous.map((item) =>
          item.id === failedChat.id
            ? failedChat
            : item,
        ),
      );

      showToast(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFiles(
    files: File[],
  ) {
    const pdfs = files.filter(
      (file) =>
        file.type ===
          "application/pdf" ||
        file.name
          .toLowerCase()
          .endsWith(".pdf"),
    );

    if (pdfs.length === 0) {
      showToast(
        "Please select a PDF document.",
      );

      return;
    }

    setUploading(true);

    try {
      for (const file of pdfs) {
        try {
          const document =
            await uploadPDF(
              file,
              settings.apiBaseUrl,
              (progress) => {
                setDocuments(
                  (previous) =>
                    previous.map(
                      (item) =>
                        item.name ===
                        file.name
                          ? {
                              ...item,
                              progress,
                            }
                          : item,
                    ),
                );
              },
            );

          upsertDocument(document);

          setDocuments(
            loadDocuments(),
          );

          setActiveDocumentId(
            document.id,
          );

          const newChat =
            createChat([
              document.id,
            ]);

          updateChat(newChat);

          setChats(loadChats());

          setActiveChatId(
            newChat.id,
          );

          showToast(
            `${document.name} uploaded successfully.`,
          );
        } catch (error) {
          const message =
            error instanceof Error
              ? error.message
              : "Document upload failed.";

          showToast(message);
        }
      }
    } finally {
      setUploading(false);
    }
  }

  function handleDeleteChat(
    chatId: string,
  ) {
    deleteChat(chatId);

    setChats(
      loadChats(),
    );

    if (
      activeChatId === chatId
    ) {
      setActiveChatId(null);
    }
  }

  function handleRenameChat(
    chatId: string,
  ) {
    const chat =
      chats.find(
        (item) =>
          item.id === chatId,
      );

    if (!chat) {
      return;
    }

    const title =
      window.prompt(
        "Rename conversation",
        chat.title,
      );

    if (
      title === null
    ) {
      return;
    }

    renameChat(
      chatId,
      title,
    );

    refreshChats();
  }

  function handleSelectChat(
    chat: ChatSession,
  ) {
    setActiveChatId(chat.id);

    const documentId =
      chat.documentIds[
        chat.documentIds.length -
          1
      ];

    if (documentId) {
      setActiveDocumentId(
        documentId,
      );
    }

    if (
      window.innerWidth < 900
    ) {
      setSidebarOpen(false);
    }
  }

  function handleSelectDocument(
    document: DocumentRecord,
  ) {
    setActiveDocumentId(
      document.id,
    );

    const relatedChat =
      chats
        .filter((chat) =>
          chat.documentIds.includes(
            document.id,
          ),
        )
        .sort(
          (a, b) =>
            b.updatedAt -
            a.updatedAt,
        )[0];

    if (relatedChat) {
      setActiveChatId(
        relatedChat.id,
      );
    }

    if (
      window.innerWidth < 900
    ) {
      setSidebarOpen(false);
    }
  }

  function handleRemoveDocument(
    documentId: string,
  ) {
    removeDocument(
      documentId,
    );

    setDocuments(
      loadDocuments(),
    );

    if (
      activeDocumentId ===
      documentId
    ) {
      setActiveDocumentId(null);
    }
  }

  const title =
    activeDocument?.name ||
    "Document Analyzer";

  const documentCount =
    documents.length;

  const chatCount =
    chats.length;

  const sidebarClass =
    sidebarCollapsed
      ? "collapsed"
      : "";

  return (
    <div
      className={[
        "app-shell",
        sidebarClass,
      ].join(" ")}
    >
      <Sidebar
        open={sidebarOpen}
        collapsed={
          sidebarCollapsed
        }
        chats={chats}
        documents={documents}
        activeChatId={
          activeChatId
        }
        activeDocumentId={
          activeDocumentId
        }
        onNewChat={
          handleNewChat
        }
        onSelectChat={
          handleSelectChat
        }
        onDeleteChat={
          handleDeleteChat
        }
        onRenameChat={
          handleRenameChat
        }
        onSelectDocument={
          handleSelectDocument
        }
        onUploadClick={() =>
          fileInputRef.current?.click()
        }
        onSettings={() =>
          setSettingsOpen(true)
        }
        onCloseMobile={() =>
          setSidebarOpen(false)
        }
      />

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button"
              onClick={() => {
                if (
                  window.innerWidth <
                  900
                ) {
                  setSidebarOpen(
                    true,
                  );
                } else {
                  setSidebarCollapsed(
                    (value) =>
                      !value,
                  );
                }
              }}
              aria-label="Toggle sidebar"
            >
              {window.innerWidth <
              900 ? (
                <Menu size={19} />
              ) : sidebarCollapsed ? (
                <PanelLeftOpen
                  size={19}
                />
              ) : (
                <PanelLeftClose
                  size={19}
                />
              )}
            </button>

            <div className="topbar-title">
              <span>
                {title}
              </span>

              {activeDocument && (
                <small>
                  PDF · Ready
                </small>
              )}
            </div>
          </div>

          <div className="topbar-right">
            <button
              className="topbar-action"
              onClick={
                handleNewChat
              }
            >
              <Plus size={17} />
              <span>
                New chat
              </span>
            </button>

            <button
              className="topbar-action search-action"
              aria-label="Search conversations"
              title="Search conversations"
            >
              <Search size={17} />
            </button>
          </div>
        </header>

        <div className="content-area">
          {!hasDocument ? (
            <div className="document-start">
              <div className="document-start-heading">
                <div className="hero-badge">
                  <div className="brand-mark">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>

                <h1>
                  Your documents,
                  <br />
                  <span>
                    intelligently analyzed.
                  </span>
                </h1>

                <p>
                  Upload a PDF and ask
                  questions using
                  grounded retrieval and
                  AI-powered document
                  understanding.
                </p>
              </div>

              <UploadZone
                documents={documents}
                uploading={uploading}
                onFiles={handleFiles}
                onRemove={
                  handleRemoveDocument
                }
              />

              <div className="quick-stats">
                <div>
                  <strong>
                    {documentCount}
                  </strong>
                  <span>
                    Documents
                  </span>
                </div>

                <div>
                  <strong>
                    {chatCount}
                  </strong>
                  <span>
                    Conversations
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <>
              <ChatWindow
                messages={
                  currentMessages
                }
                loading={loading}
                onExampleQuestion={
                  (question) =>
                    handleSend(
                      question,
                    )
                }
              />

              <ChatInput
                value={input}
                disabled={
                  !activeDocument
                }
                loading={loading}
                activeDocument={
                  activeDocument
                }
                onChange={setInput}
                onSubmit={() =>
                  handleSend()
                }
                onAttach={() =>
                  fileInputRef.current?.click()
                }
              />
            </>
          )}
        </div>
      </main>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(event) => {
          handleFiles(
            Array.from(
              event.target.files ||
                [],
            ),
          );

          event.target.value = "";
        }}
      />

      <SettingsModal
        open={settingsOpen}
        settings={settings}
        onChange={setSettings}
        onClose={() =>
          setSettingsOpen(false)
        }
      />
    </div>
  );
}

function createChatTitle(
  question: string,
): string {
  const normalized =
    question
      .replace(/\s+/g, " ")
      .trim();

  if (
    normalized.length <= 42
  ) {
    return normalized;
  }

  return `${normalized.slice(
    0,
    42,
  )}…`;
}

function showToast(
  message: string,
) {
  window.dispatchEvent(
    new CustomEvent(
      "docurag:toast",
      {
        detail: message,
      },
    ),
  );
}

export default App;
