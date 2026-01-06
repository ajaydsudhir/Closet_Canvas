import { useEffect, useMemo, useRef } from "react";
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
import { useSessionStatus } from "../hooks/useSessionStatus";
import type { ClipSource } from "../types/clips";
import type { UserData } from "../types/user";
import StatusIndicator from "./StatusIndicator";

type UserSessionWorkspaceProps = {
  sessionId: string;
  user: UserData;
  onClipsReady?: () => void;
  onCameraError?: () => void;
};

const UserSessionWorkspace = ({
  sessionId,
  user,
  onClipsReady,
  onCameraError,
}: UserSessionWorkspaceProps) => {
  const truncatedId = useMemo(() => sessionId.slice(0, 8), [sessionId]);
  const stopRecordingRef = useRef<(() => void) | null>(null);
  const uploadedClipsCountRef = useRef(0);

  // WebSocket status tracking
  const { status, message } = useSessionStatus({
    sessionId,
    enabled: true,
  });

  // Auto-stop recording when status reaches 'smpl' or 'complete'
  useEffect(() => {
    if ((status === 'smpl' || status === 'complete') && stopRecordingRef.current) {
      console.log(`Auto-stopping recording due to status: ${status}`);
      stopRecordingRef.current();
      stopRecordingRef.current = null;
    }
  }, [status]);

  const handleLiveSegment = (segment: LiveRecorderSegment) => {
    const filename = `live_${segment.startedAt.replace(
      /[:.]/g,
      "",
    )}_${segment.sequence}.webm`;
    const file = new File([segment.blob], filename, {
      type: segment.mimeType || "video/webm",
    });
    void enqueueFileForUpload(file, "live", {
      mime_type: segment.mimeType || file.type,
      sequence_no: segment.sequence,
      capture_started_at: segment.startedAt,
      duration_ms: segment.durationMs,
    });
  };

  const enqueueFileForUpload = async (
    file: File,
    source: ClipSource,
    metadata?: ClipMetadata,
  ) => {
    // Don't upload more than 5 clips
    if (uploadedClipsCountRef.current >= 5) {
      console.log("Already uploaded 5 clips, skipping further uploads");
      return;
    }

    try {
      const uploadInfo = await requestClipUpload({
        sessionId,
        filename: file.name,
        contentType: file.type || "application/octet-stream",
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

      // Increment uploaded clips counter
      uploadedClipsCountRef.current += 1;
      console.log(`Uploaded ${uploadedClipsCountRef.current} clips`);

      // When we reach 5 clips, stop recording and transition
      if (uploadedClipsCountRef.current === 5) {
        console.log("5 clips uploaded, stopping recording and transitioning to processing");
        
        // Stop recording immediately
        if (stopRecordingRef.current) {
          stopRecordingRef.current();
          stopRecordingRef.current = null;
        }
        
        // Notify parent to transition
        if (onClipsReady) {
          onClipsReady();
        }
      }
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

      {/* Status Indicator - Top Right */}
      <div className="absolute right-6 top-6 z-20">
        <StatusIndicator status={status || "idle"} message={message} />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6">
        {/* Camera View */}
        <div className="w-full max-w-4xl space-y-6">
          <div className="text-center">
            <h1 className="momo-trust-display-regular text-3xl text-white">
              Welcome, {user.name}
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Session: <span className="font-mono">{truncatedId}</span>
            </p>
          </div>

          <LiveCameraView 
            onSegmentReady={handleLiveSegment} 
            stopRecordingRef={stopRecordingRef}
            onCameraError={onCameraError}
          />
        </div>
      </div>
    </main>
  );
};

const LiveCameraView = ({
  onSegmentReady,
  stopRecordingRef,
  onCameraError,
}: {
  onSegmentReady: (segment: LiveRecorderSegment) => void;
  stopRecordingRef: React.MutableRefObject<(() => void) | null>;
  onCameraError?: () => void;
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

  // Auto-start camera on mount
  useEffect(() => {
    if (isSupported && !isRecording && !stream) {
      start();
    }
  }, [isSupported]);

  // Update stopRecordingRef when stop function changes
  useEffect(() => {
    if (isRecording) {
      stopRecordingRef.current = stop;
    } else {
      stopRecordingRef.current = null;
    }
  }, [isRecording, stop]);

  return (
    <div className="space-y-6 rounded-3xl border border-white/10 bg-slate-900/60 p-8 shadow-[0_20px_35px_rgba(2,6,23,0.55)]">
      {/* Camera Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {!isRecording && stream && (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              className="h-10 max-w-[200px] rounded-full border border-white/10 bg-slate-900/40 px-4 text-sm text-slate-200 outline-none focus:border-white/20"
            >
              <option value="">Default Camera</option>
              {devices.map((device) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="flex items-center gap-3">
          {!isRecording && stream && (
            <button
              type="button"
              onClick={start}
              className="inline-flex h-10 items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-500/90 px-6 text-sm font-semibold text-white shadow-[0_12px_25px_rgba(16,185,129,0.35)] transition hover:-translate-y-0.5 hover:bg-emerald-500"
            >
              Start Recording
            </button>
          )}
          {isRecording && (
            <div className="inline-flex h-10 items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-500/90 px-6 text-sm font-semibold text-white">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-white"></span>
              Recording...
            </div>
          )}
        </div>
      </div>

      {/* Video Display */}
      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
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
            {isRecording && (
              <p className="text-center text-sm text-slate-400">
                Recording in progress... clips are being sent every ~3 seconds
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-6 py-20 text-center">
            <p className="text-base font-semibold uppercase tracking-[0.35em] text-slate-300">
              {error ? "Camera Error" : "Initializing Camera"}
            </p>
            <p className="text-sm text-slate-400">
              {error || "Please allow camera access to continue..."}
            </p>
            {error && onCameraError && (
              <div className="space-y-3">
                <p className="text-sm text-rose-300">
                  Unable to access your camera. Please try again later.
                </p>
                <button
                  type="button"
                  onClick={onCameraError}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-6 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60"
                >
                  Return to Login
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const buildClipMetadata = (
  source: ClipSource,
  extra?: ClipMetadata,
): ClipUploadMetadata => ({
  source,
  ...(extra ?? {}),
});

export default UserSessionWorkspace;