import { CopySimple } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  notifyClipUploaded,
  requestClipUpload,
  uploadFileToPresignedUrl,
  type ClipMetadata,
  type ClipUploadMetadata,
} from "../api/capture";
import {
  useLiveRecorder,
  type LiveRecorderSegment,
} from "../hooks/useLiveRecorder";
import type { ClipRow, ClipSource } from "../types/clips";
import SessionClipsBoard from "./SessionClipsBoard";

type SessionWorkspaceProps = {
  sessionId: string;
};

const SessionWorkspace = ({ sessionId }: SessionWorkspaceProps) => {
  const [clips, setClips] = useState<ClipRow[]>([]);
  const [mode, setMode] = useState<"clips" | "live">("clips");
  const truncatedId = useMemo(() => sessionId.slice(0, 8), [sessionId]);

  const handleAddFiles = (files: FileList) => {
    Array.from(files).forEach((file) => enqueueFileForUpload(file, "manual"));
  };

  const handleLiveSegment = (segment: LiveRecorderSegment) => {
    const filename = `live_${segment.startedAt.replace(
      /[:.]/g,
      "",
    )}_${segment.sequence}.webm`;
    const file = new File([segment.blob], filename, {
      type: segment.mimeType || "video/webm",
    });
    enqueueFileForUpload(file, "live", {
      mime_type: segment.mimeType || file.type,
      sequence_no: segment.sequence,
      capture_started_at: segment.startedAt,
      duration_ms: segment.durationMs,
    });
  };

  const enqueueFileForUpload = (
    file: File,
    source: ClipSource,
    metadata?: ClipMetadata,
  ) => {
    const localId = generateLocalId();
    const clip: ClipRow = {
      id: localId,
      title:
        file.name ||
        (source === "live"
          ? `Live segment ${new Date().toLocaleTimeString()}`
          : "Untitled Clip"),
      status: "Queued",
      source,
    };
    setClips((prev) => [...prev, clip]);
    void handleUploadFlow(localId, file, source, metadata);
  };

  const handleUploadFlow = async (
    localId: string,
    file: File,
    source: ClipSource,
    metadata?: ClipMetadata,
  ) => {
    try {
      updateClip(localId, { status: "Uploading", message: undefined });
      const uploadInfo = await requestClipUpload({
        sessionId,
        filename: file.name,
        contentType: file.type || "application/octet-stream",
      });

      updateClip(localId, {
        remoteId: uploadInfo.clip_id,
        status: "Uploading",
      });

      await uploadFileToPresignedUrl({
        url: uploadInfo.upload_url,
        file,
        contentType: file.type || "application/octet-stream",
      });

      await notifyClipUploaded({
        sessionId,
        clipId: uploadInfo.clip_id,
        metadata: buildClipMetadata(source, metadata),
      });

      updateClip(localId, { status: "Uploaded" });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      updateClip(localId, { status: "Error", message });
    }
  };

  const updateClip = (localId: string, changes: Partial<ClipRow>) => {
    setClips((prev) =>
      prev.map((clip) =>
        clip.id === localId ? { ...clip, ...changes } : clip,
      ),
    );
  };

  const handleCopySessionId = async () => {
    try {
      await navigator.clipboard.writeText(sessionId);
    } catch (error) {
      console.warn("Unable to copy session id", error);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 px-6 py-10 text-slate-100">
      <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

      <div className="relative z-10 mx-auto max-w-5xl space-y-8">
        <header className="flex items-center gap-4 rounded-3xl border border-white/10 bg-slate-900/70 px-6 py-4 shadow-[0_20px_35px_rgba(2,6,23,0.55)]">
          <img
            src="/logo.svg"
            alt="Closet Canvas"
            className="h-10 w-10 rounded-2xl bg-white/5 p-2"
          />
          <p className="momo-trust-display-regular text-3xl text-white">
            Closet Canvas
          </p>
        </header>

        <div className="flex items-center justify-center gap-4">
          <TabButton
            label="Clips"
            active={mode === "clips"}
            onClick={() => setMode("clips")}
          />
          <TabButton
            label="Live"
            active={mode === "live"}
            onClick={() => setMode("live")}
          />
        </div>

        {mode === "clips" ? (
          <SessionClipsBoard
            sessionId={sessionId}
            truncatedId={truncatedId}
            clips={clips}
            onAddFiles={handleAddFiles}
            onCopySessionId={handleCopySessionId}
          />
        ) : (
          <LivePanel
            sessionId={sessionId}
            truncatedId={truncatedId}
            onCopySessionId={handleCopySessionId}
            onSegmentReady={handleLiveSegment}
          />
        )}
      </div>
    </main>
  );
};

const TabButton = ({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`inline-flex items-center rounded-full px-6 py-2 text-sm font-semibold transition ${
      active
        ? "bg-white text-slate-900 shadow-[0_10px_25px_rgba(15,23,42,0.35)]"
        : "bg-white/10 text-slate-300 hover:bg-white/20"
    }`}
  >
    {label}
  </button>
);

const LivePanel = ({
  sessionId,
  truncatedId,
  onCopySessionId,
  onSegmentReady,
}: {
  sessionId: string;
  truncatedId: string;
  onCopySessionId: () => void;
  onSegmentReady: (segment: LiveRecorderSegment) => void;
}) => {
  const {
    isSupported,
    isRecording,
    error,
    start,
    stop,
    stream,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
  } = useLiveRecorder({
    timesliceMs: 3000,
    onSegment: onSegmentReady,
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <section className="space-y-6 rounded-3xl border border-white/10 bg-slate-900/60 p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-shrink-0 items-center gap-2">
          <span className="inline-flex h-10 items-center rounded-full border border-white/10 bg-slate-900/40 px-4 font-mono text-lg tracking-wide text-white">
            {truncatedId}
          </span>
          <button
            type="button"
            onClick={onCopySessionId}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-700/80 bg-slate-900/60 text-slate-200 transition hover:border-slate-500"
            aria-label={`Copy session id ${sessionId}`}
          >
            <CopySimple size={18} weight="bold" />
          </button>
        </div>
        <div className="flex items-center gap-3">
          {!isRecording && (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              className="h-10 max-w-[200px] rounded-full border border-white/10 bg-slate-900/40 px-4 text-center text-sm text-slate-200 outline-none focus:border-white/20 appearance-none"
              style={{ backgroundImage: "none" }}
            >
              <option value="">Default Camera</option>
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={start}
            disabled={!isSupported || isRecording}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-500/90 px-5 text-sm font-semibold text-white shadow-[0_12px_25px_rgba(16,185,129,0.35)] transition hover:-translate-y-0.5 hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRecording ? "Recording…" : "Start Broadcast"}
          </button>
          <button
            type="button"
            onClick={stop}
            disabled={!isRecording}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-rose-400/40 bg-rose-500/20 px-5 text-sm font-semibold text-rose-100 transition hover:-translate-y-0.5 hover:bg-rose-500/40 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Stop
          </button>
        </div>
      </div>
      <div className="rounded-2xl border border-dashed border-white/20 bg-slate-950/50 p-4 text-slate-300">
        {stream ? (
          <div className="space-y-3">
            <div className="overflow-hidden rounded-2xl border border-white/10">
              <div className="aspect-video bg-black">
                <video
                  ref={videoRef}
                  className="h-full w-full object-contain"
                  autoPlay
                  muted
                  playsInline
                />
              </div>
            </div>
            <p className="text-sm text-slate-400">
              {isRecording
                ? "Broadcasting… segments arrive every ~3 seconds."
                : "Preview ready. Start broadcast to begin sending segments."}
            </p>
          </div>
        ) : (
          <div className="space-y-3 text-center">
            <p className="text-base font-semibold uppercase tracking-[0.35em]">
              Live Capture
            </p>
            <p className="text-sm text-slate-400">
              {error
                ? error
                : isSupported
                  ? "Recording sends a new clip every ~3 seconds to the queue."
                  : "Live capture is not supported in this browser."}
            </p>
          </div>
        )}
      </div>
    </section>
  );
};

const buildClipMetadata = (
  source: ClipSource,
  extra?: ClipMetadata,
): ClipUploadMetadata => ({
  source,
  ...(extra ?? {}),
});

const generateLocalId = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random()}`;

export default SessionWorkspace;
