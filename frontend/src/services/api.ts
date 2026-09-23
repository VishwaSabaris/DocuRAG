import type {
  AskResponse,
  DocumentRecord,
  UploadResponse,
} from "../types/chat";

function normalizeBaseUrl(
  baseUrl: string,
): string {
  return baseUrl.replace(/\/+$/, "");
}

export async function uploadPDF(
  file: File,
  baseUrl: string,
  onProgress?: (progress: number) => void,
): Promise<DocumentRecord> {
  if (
    file.type !== "application/pdf" &&
    !file.name.toLowerCase().endsWith(".pdf")
  ) {
    throw new Error(
      "Only PDF documents are supported.",
    );
  }

  const formData = new FormData();

  formData.append("file", file);

  const url = `${normalizeBaseUrl(
    baseUrl,
  )}/documents/upload`;

  onProgress?.(5);

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  onProgress?.(85);

  if (!response.ok) {
    let message =
      "Document upload failed.";

    try {
      const data = await response.json();

      message =
        data.detail ||
        data.message ||
        message;
    } catch {
      // Keep fallback message.
    }

    throw new Error(message);
  }

  const data =
    (await response.json()) as UploadResponse;

  onProgress?.(100);

  const id =
    data.document_id ||
    data.id;

  if (!id) {
    throw new Error(
      "Backend upload response did not contain a document ID.",
    );
  }

  return {
    id,
    name:
      data.filename ||
      data.name ||
      file.name,
    size: file.size,
    mimeType: file.type || "application/pdf",
    uploadedAt: Date.now(),
    status: "ready",
    progress: 100,
  };
}

export async function askDocument(
  documentId: string,
  query: string,
  baseUrl: string,
): Promise<AskResponse> {
  const url =
    `${normalizeBaseUrl(baseUrl)}` +
    `/documents/${encodeURIComponent(
      documentId,
    )}/ask`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
    }),
  });

  if (!response.ok) {
    let message =
      "Unable to get an answer.";

    try {
      const data = await response.json();

      message =
        data.detail ||
        data.message ||
        message;
    } catch {
      // Keep fallback.
    }

    throw new Error(message);
  }

  return (await response.json()) as AskResponse;
}

/*
 * Optional future API functions.
 *
 * Keep these here when the backend later supports
 * server-side conversation persistence.
 */

export async function fetchChatHistory(
  baseUrl: string,
): Promise<unknown> {
  const response = await fetch(
    `${normalizeBaseUrl(
      baseUrl,
    )}/chats`,
  );

  if (!response.ok) {
    throw new Error(
      "Unable to fetch chat history.",
    );
  }

  return response.json();
}

export async function deleteRemoteDocument(
  documentId: string,
  baseUrl: string,
): Promise<void> {
  const response = await fetch(
    `${normalizeBaseUrl(
      baseUrl,
    )}/documents/${encodeURIComponent(
      documentId,
    )}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    throw new Error(
      "Unable to delete document.",
    );
  }
}
