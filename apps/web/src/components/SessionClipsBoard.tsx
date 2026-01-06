import { useMemo, useRef } from "react";
import { CheckCircleIcon, CopySimpleIcon } from "@phosphor-icons/react";
import type { ClipRow, ClipStatus } from "../types/clips";

type SessionClipsBoardProps = {
  sessionId: string;
  truncatedId: string;
  clips: ClipRow[];
  onAddFiles: (files: FileList) => void;
  onCopySessionId: () => void;
};

const SessionClipsBoard = ({
  sessionId,
  truncatedId,
  clips,
  onAddFiles,
  onCopySessionId,
}: SessionClipsBoardProps) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleInvokePicker = () => fileInputRef.current?.click();

  const handleFilesSelected = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target;
    if (files?.length) {
      onAddFiles(files);
      event.target.value = "";
    }
  };

  const uploadedCount = useMemo(
    () => clips.filter((clip) => clip.status === "Uploaded").length,
    [clips],
  );

  return (
    <section className="space-y-6 rounded-3xl border border-white/10 bg-slate-900/60 p-8">
      <div className="space-y-3">
        <p className="text-xs uppercase tracking-[0.35em] text-slate-400">
          Session
        </p>
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
              <CopySimpleIcon size={18} weight="bold" />
            </button>
          </div>
          <button
            type="button"
            onClick={handleInvokePicker}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-indigo-400/40 bg-indigo-500/90 px-5 text-sm font-semibold text-white shadow-[0_12px_25px_rgba(79,70,229,0.35)] transition hover:-translate-y-0.5 hover:bg-indigo-500"
          >
            <span className="text-base leading-none">+</span>
            Add Clips
          </button>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={handleFilesSelected}
      />

      <section className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-[0_30px_60px_rgba(2,6,23,0.65)] backdrop-blur">
        <div className="overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-y-2">
            <thead>
              <tr className="text-left text-sm uppercase tracking-[0.2em] text-slate-400">
                <th className="pb-2">Clip Title</th>
                <th className="pb-2 text-right">Status</th>
              </tr>
            </thead>
            <tbody>
              {clips.length === 0 ? (
                <tr>
                  <td
                    colSpan={2}
                    className="rounded-2xl bg-slate-900/60 px-4 py-8 text-center text-sm text-slate-400"
                  >
                    No clips yet. Use “Add Clips” to queue uploads for this
                    session.
                  </td>
                </tr>
              ) : (
                clips.map((clip) => (
                  <tr
                    key={clip.id}
                    className="rounded-2xl bg-slate-900/60 text-sm text-white"
                  >
                    <td className="rounded-l-2xl px-4 py-4 font-medium">
                      {clip.title}
                    </td>
                    <td className="rounded-r-2xl px-4 py-4 text-right">
                      <StatusBadge status={clip.status} />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="flex flex-wrap items-center justify-center gap-10 text-sm text-slate-300">
        <p className="text-center">
          <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
            Clips Added
          </span>
          <span className="mt-1 block text-3xl font-semibold text-white">
            {clips.length}
          </span>
        </p>
        <p className="text-center">
          <span className="text-xs uppercase tracking-[0.3em] text-slate-500">
            Clips Uploaded
          </span>
          <span className="mt-1 block text-3xl font-semibold text-white">
            {uploadedCount}
          </span>
        </p>
      </div>
    </section>
  );
};

const StatusBadge = ({ status }: { status: ClipStatus }) => {
  const base =
    "inline-flex items-center gap-3 rounded-full px-4 py-2 text-sm font-semibold";
  const style =
    status === "Uploaded"
      ? "bg-emerald-400/15 text-emerald-200"
      : status === "Uploading"
        ? "bg-indigo-400/15 text-indigo-200"
        : status === "Error"
          ? "bg-rose-400/15 text-rose-200"
          : "bg-amber-400/15 text-amber-200";
  const iconStyle =
    status === "Uploaded"
      ? "text-emerald-300"
      : status === "Uploading"
        ? "text-indigo-300"
        : status === "Error"
          ? "text-rose-300"
          : "text-amber-300";

  return (
    <span className={`${base} ${style}`}>
      <CheckCircleIcon
        size={18}
        weight={status === "Uploaded" ? "fill" : "regular"}
        className={iconStyle}
      />
      {status}
    </span>
  );
};

export default SessionClipsBoard;
