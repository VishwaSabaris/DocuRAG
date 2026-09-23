import type {
  AppSettings,
  ChatSession,
  DocumentRecord,
} from "../types/chat";

const CHAT_STORAGE_KEY = "docurag.chats.v1";
const DOCUMENT_STORAGE_KEY = "docurag.documents.v1";
const SETTINGS_STORAGE_KEY = "docurag.settings.v1";

const DEFAULT_SETTINGS: AppSettings = {
  theme: "dark",
  apiBaseUrl: "http://localhost:8000",
};

function safelyParse<T>(value: string | null, fallback: T): T {
  if (!value) {
    return fallback;
  }

  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

export function loadChats(): ChatSession[] {
  return safelyParse<ChatSession[]>(
    localStorage.getItem(CHAT_STORAGE_KEY),
    [],
  );
}

export function saveChats(chats: ChatSession[]): void {
  localStorage.setItem(
    CHAT_STORAGE_KEY,
    JSON.stringify(chats),
  );
}

export function loadDocuments(): DocumentRecord[] {
  return safelyParse<DocumentRecord[]>(
    localStorage.getItem(DOCUMENT_STORAGE_KEY),
    [],
  );
}

export function saveDocuments(
  documents: DocumentRecord[],
): void {
  localStorage.setItem(
    DOCUMENT_STORAGE_KEY,
    JSON.stringify(documents),
  );
}

export function loadSettings(): AppSettings {
  return safelyParse<AppSettings>(
    localStorage.getItem(SETTINGS_STORAGE_KEY),
    DEFAULT_SETTINGS,
  );
}

export function saveSettings(
  settings: AppSettings,
): void {
  localStorage.setItem(
    SETTINGS_STORAGE_KEY,
    JSON.stringify(settings),
  );
}

export function createId(prefix = "id"): string {
  return `${prefix}_${Date.now()}_${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

export function createChat(
  documentIds: string[] = [],
): ChatSession {
  const now = Date.now();

  return {
    id: createId("chat"),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    documentIds,
    messages: [],
  };
}

export function updateChat(
  chat: ChatSession,
): void {
  const chats = loadChats();

  const updated = chats.some(
    (item) => item.id === chat.id,
  )
    ? chats.map((item) =>
        item.id === chat.id ? chat : item,
      )
    : [chat, ...chats];

  saveChats(
    updated.sort(
      (a, b) => b.updatedAt - a.updatedAt,
    ),
  );
}

export function deleteChat(
  chatId: string,
): void {
  const chats = loadChats().filter(
    (chat) => chat.id !== chatId,
  );

  saveChats(chats);
}

export function renameChat(
  chatId: string,
  title: string,
): void {
  const chats = loadChats().map((chat) =>
    chat.id === chatId
      ? {
          ...chat,
          title:
            title.trim() || "New conversation",
          updatedAt: Date.now(),
        }
      : chat,
  );

  saveChats(chats);
}

export function upsertDocument(
  document: DocumentRecord,
): void {
  const documents = loadDocuments();

  const exists = documents.some(
    (item) => item.id === document.id,
  );

  const next = exists
    ? documents.map((item) =>
        item.id === document.id
          ? document
          : item,
      )
    : [document, ...documents];

  saveDocuments(
    next.sort(
      (a, b) => b.uploadedAt - a.uploadedAt,
    ),
  );
}

export function removeDocument(
  documentId: string,
): void {
  saveDocuments(
    loadDocuments().filter(
      (document) =>
        document.id !== documentId,
    ),
  );
}
