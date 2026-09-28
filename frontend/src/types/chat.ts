export interface SourceReference {
  chunk_id?: string;
  document_id?: string;
  page_number?: number | null;
  chunk_index?: number;
  score?: number;
  rerank_score?: number | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  sources?: SourceReference[];
  isError?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  documentIds: string[];
  messages: ChatMessage[];
}

export interface DocumentRecord {
  id: string;
  name: string;
  size: number;
  mimeType: string;
  uploadedAt: number;
  status: "uploading" | "processing" | "ready" | "error";
  progress?: number;
}

export interface AppSettings {
  theme: "dark" | "light";
  apiBaseUrl: string;
}

export interface AskResponse {
  answer: string;
  sources?: SourceReference[];
}

export interface UploadResponse {
  id: string;
  document_id?: string;
  filename?: string;
  name?: string;
  status?: string;
  message?: string;
}
