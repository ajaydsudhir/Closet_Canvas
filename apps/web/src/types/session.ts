export type SessionStatus = 
  | "idle"
  | "recording"
  | "gating"      // Video quality check - 33% progress
  | "smpl"        // Body pose estimation - 67% progress
  | "finishing"   // Final processing - 90% progress
  | "recommending"
  | "complete"
  | "error";

export type SessionStatusMessage = {
  type: "status";
  sessionId: string;
  status: SessionStatus;
  message?: string;
  progress?: number; // 0-100
  timestamp: string;
};

export type SessionErrorMessage = {
  type: "error";
  sessionId: string;
  error: string;
  timestamp: string;
};

export type WebSocketMessage = SessionStatusMessage | SessionErrorMessage;
