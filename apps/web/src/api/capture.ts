const API_ROOT = "/api";

export type CreateClipResponse = {
  clip_id: string;
  object_key: string;
  upload_url: string;
  expires_at: string;
};

export async function createSession(): Promise<{ session_id: string }> {
  const response = await fetch(`${API_ROOT}/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.status}`);
  }

  return response.json();
}

export async function requestClipUpload({
  sessionId,
  filename,
  contentType,
}: {
  sessionId: string;
  filename: string;
  contentType: string;
}): Promise<CreateClipResponse> {
  const response = await fetch(`${API_ROOT}/v1/sessions/${sessionId}/clips`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, content_type: contentType }),
  });

  if (!response.ok) {
    throw new Error(`Failed to request upload: ${response.status}`);
  }

  return response.json();
}

export type ClipMetadata = {
  mime_type?: string;
  sequence_no?: number;
  capture_started_at?: string;
  duration_ms?: number;
};

export type ClipUploadMetadata = ClipMetadata & {
  source: "manual" | "live";
};

export async function notifyClipUploaded({
  sessionId,
  clipId,
  metadata,
}: {
  sessionId: string;
  clipId: string;
  metadata: ClipUploadMetadata;
}): Promise<void> {
  const response = await fetch(
    `${API_ROOT}/v1/sessions/${sessionId}/clips/${clipId}/uploaded`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata: metadata ?? {} }),
    },
  );

  if (!response.ok) {
    throw new Error(`Failed to notify upload: ${response.status}`);
  }
}

export async function uploadFileToPresignedUrl({
  url,
  file,
  contentType,
}: {
  url: string;
  file: File;
  contentType: string;
}): Promise<void> {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": contentType },
    body: file,
  });

  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`);
  }
}
