import type { SessionStatus } from "../types/session";

type StatusIndicatorProps = {
  status: SessionStatus;
  message?: string;
};

const StatusIndicator = ({ status, message }: StatusIndicatorProps) => {
  const getStatusConfig = () => {
    switch (status) {
      case "recording":
        return {
          label: "Recording",
          color: "emerald",
          bgColor: "bg-emerald-500/20",
          borderColor: "border-emerald-500/40",
          dotColor: "bg-emerald-400",
          textColor: "text-emerald-300",
          progress: undefined,
        };
      case "gating":
        return {
          label: "Processing",
          color: "amber",
          bgColor: "bg-amber-500/20",
          borderColor: "border-amber-500/40",
          dotColor: "bg-amber-400",
          textColor: "text-amber-300",
          progress: 33,
        };
      case "smpl":
        return {
          label: "SMPL Analysis",
          color: "amber",
          bgColor: "bg-amber-500/20",
          borderColor: "border-amber-500/40",
          dotColor: "bg-amber-400",
          textColor: "text-amber-300",
          progress: 67,
        };
      case "finishing":
        return {
          label: "Finishing Up",
          color: "blue",
          bgColor: "bg-blue-500/20",
          borderColor: "border-blue-500/40",
          dotColor: "bg-blue-400",
          textColor: "text-blue-300",
          progress: 90,
        };
      case "recommending":
        return {
          label: "Recommending",
          color: "violet",
          bgColor: "bg-violet-500/20",
          borderColor: "border-violet-500/40",
          dotColor: "bg-violet-400",
          textColor: "text-violet-300",
          progress: 95,
        };
      case "complete":
        return {
          label: "Complete",
          color: "emerald",
          bgColor: "bg-emerald-500/20",
          borderColor: "border-emerald-500/40",
          dotColor: "bg-emerald-400",
          textColor: "text-emerald-300",
          progress: 100,
        };
      case "error":
        return {
          label: "Error",
          color: "rose",
          bgColor: "bg-rose-500/20",
          borderColor: "border-rose-500/40",
          dotColor: "bg-rose-400",
          textColor: "text-rose-300",
          progress: undefined,
        };
      default:
        return {
          label: "Idle",
          color: "slate",
          bgColor: "bg-slate-500/20",
          borderColor: "border-slate-500/40",
          dotColor: "bg-slate-400",
          textColor: "text-slate-300",
          progress: undefined,
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border ${config.borderColor} ${config.bgColor} px-4 py-2 backdrop-blur-sm shadow-lg`}
    >
      {/* Blinking dot */}
      <span className="relative flex h-2 w-2">
        <span
          className={`absolute inline-flex h-full w-full animate-ping rounded-full ${config.dotColor} opacity-75`}
        />
        <span className={`relative inline-flex h-2 w-2 rounded-full ${config.dotColor}`} />
      </span>

      {/* Status text */}
      <div className="flex items-center gap-2">
        <span className={`text-sm font-semibold ${config.textColor}`}>
          {config.label}
        </span>
        {config.progress !== undefined && (
          <span className={`text-xs ${config.textColor}`}>
            {config.progress}%
          </span>
        )}
      </div>

      {/* Optional message */}
      {message && (
        <span className="text-xs text-slate-400">
          • {message}
        </span>
      )}
    </div>
  );
};

export default StatusIndicator;
