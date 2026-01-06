import { useCallback, useEffect, useRef, useState } from "react";

export type LiveRecorderSegment = {
  blob: Blob;
  sequence: number;
  startedAt: string;
  durationMs: number;
  mimeType: string;
};

type UseLiveRecorderOptions = {
  timesliceMs?: number;
  onSegment: (segment: LiveRecorderSegment) => void;
};

export const useLiveRecorder = ({
  timesliceMs = 3000,
  onSegment,
}: UseLiveRecorderOptions) => {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewStream, setPreviewStream] = useState<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const startTimestampRef = useRef<number | null>(null);
  const sequenceRef = useRef(0);

  const preferredMimeTypes = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm;codecs=h264",
    "video/webm",
  ];

  const isSupported =
    typeof window !== "undefined" &&
    typeof window.MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia;

  useEffect(() => {
    if (!isSupported) return;

    const getDevices = async () => {
      try {
        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = deviceList.filter(
          (device) => device.kind === "videoinput",
        );
        setDevices(videoDevices);
      } catch (error) {
        console.warn("Error enumerating devices:", error);
      }
    };

    getDevices();
    navigator.mediaDevices.addEventListener("devicechange", getDevices);
    return () => {
      navigator.mediaDevices.removeEventListener("devicechange", getDevices);
    };
  }, [isSupported]);

  const cleanup = useCallback(() => {
    recorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    startTimestampRef.current = null;
    sequenceRef.current = 0;
    setPreviewStream(null);
    setIsRecording(false);
  }, []);

  const start = useCallback(async () => {
    if (!isSupported || isRecording) return;
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId } }
          : true,
      });
      streamRef.current = stream;
      setPreviewStream(stream);

      // If we didn't have labels before (permission issue), refresh devices now that we have a stream
      const tracks = stream.getVideoTracks();
      if (tracks.length > 0) {
        const label = tracks[0].label;
        if (label) {
          // Refreshing devices to ensure we have labels
          navigator.mediaDevices.enumerateDevices().then((deviceList) => {
            setDevices(
              deviceList.filter((device) => device.kind === "videoinput"),
            );
          });
        }
      }

      const mimeType = preferredMimeTypes.find((type) =>
        MediaRecorder.isTypeSupported(type),
      );
      if (!mimeType) {
        console.warn(
          "[useLiveRecorder] No supported MediaRecorder mime type found",
        );
        setError(
          "No supported video compression format found. Please try a different browser (Chrome, Firefox, etc).",
        );
        cleanup();
        return;
      }

      const recorder = new MediaRecorder(stream, {
        mimeType,
      });
      startTimestampRef.current = Date.now();
      sequenceRef.current = 0;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          const sequence = sequenceRef.current++;
          const startedAtMs =
            (startTimestampRef.current ?? Date.now()) + sequence * timesliceMs;
          onSegment({
            blob: event.data,
            sequence,
            startedAt: new Date(startedAtMs).toISOString(),
            durationMs: timesliceMs,
            mimeType,
          });
        }
      };
      recorder.onerror = (event) => {
        setError(event.error?.message ?? "Recorder error");
        cleanup();
      };
      recorder.onstop = () => {
        cleanup();
      };

      recorder.start(timesliceMs);
      recorderRef.current = recorder;
      setIsRecording(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to start";
      setError(message);
      cleanup();
    }
  }, [
    cleanup,
    isRecording,
    isSupported,
    onSegment,
    selectedDeviceId,
    timesliceMs,
  ]);

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    } else {
      cleanup();
    }
  }, [cleanup]);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    isSupported,
    isRecording,
    error,
    start,
    stop,
    stream: previewStream,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
  };
};
