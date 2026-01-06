export type ClipStatus = "Queued" | "Uploading" | "Uploaded" | "Error";

export type ClipSource = "manual" | "live";

export type ClipRow = {
  id: string;
  remoteId?: string;
  title: string;
  status: ClipStatus;
  message?: string;
  source?: ClipSource;
};
