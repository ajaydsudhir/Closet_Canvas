import { CoatHanger, SignOut, Sparkle } from "@phosphor-icons/react";
import type { SessionStatus } from "../types/session";

type ProcessingScreenProps = {
  status: SessionStatus;
  message?: string;
  progress?: number;
  onLogout?: () => void;
  onRestart?: () => void;
};

// Sample fashion image URLs (using placeholder service)
const FASHION_IMAGES = [
  "https://images.unsplash.com/photo-1483985988355-763728e1935b?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1445205170230-053b83016050?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1525507119028-ed4c629a60a3?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1467043237213-65f2da53396f?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1509631179647-0177331693ae?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1582418702059-97ebafb35d09?w=400&h=600&fit=crop",
  "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=600&fit=crop",
];

const ProcessingScreen = ({ status, message, progress, onLogout, onRestart }: ProcessingScreenProps) => {
  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    } else {
      window.location.href = '/';
    }
  };

  const handleRestart = () => {
    if (onRestart) {
      onRestart();
    } else {
      window.location.href = '/';
    }
  };

  const getStatusInfo = () => {
    switch (status) {
      case "error":
        return {
          title: "Processing Error",
          subtitle: "Something went wrong during processing",
          icon: <CoatHanger size={48} weight="bold" className="text-rose-400" />,
          progressValue: 0,
        };
      case "gating":
        return {
          title: "Quality Check",
          subtitle: "Validating video quality...",
          icon: <CoatHanger size={48} weight="bold" className="animate-pulse" />,
          progressValue: 33,
        };
      case "smpl":
        return {
          title: "Body Analysis",
          subtitle: "Analyzing your body measurements and pose...",
          icon: <CoatHanger size={48} weight="bold" className="animate-bounce" />,
          progressValue: 67,
        };
      case "finishing":
        return {
          title: "Finishing Up",
          subtitle: "Finalizing your analysis...",
          icon: <Sparkle size={48} weight="fill" className="animate-spin" />,
          progressValue: 90,
        };
      case "recommending":
        return {
          title: "Finding Your Perfect Fit",
          subtitle: "Recommending clothes tailored just for you...",
          icon: <Sparkle size={48} weight="fill" className="animate-pulse" />,
          progressValue: 95,
        };
      default:
        return {
          title: "Processing",
          subtitle: "Please wait...",
          icon: <CoatHanger size={48} weight="bold" />,
          progressValue: progress || 0,
        };
    }
  };

  const statusInfo = getStatusInfo();
  const displayProgress = progress !== undefined ? progress : statusInfo.progressValue;

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

      {/* Logout button */}
      <div className="absolute right-6 top-6 z-20">
        <button
          type="button"
          onClick={handleLogout}
          className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60 backdrop-blur-sm"
        >
          <SignOut size={18} weight="bold" />
          <span>Logout</span>
        </button>
      </div>

      <div className="relative z-10 flex min-h-screen items-center px-6 lg:px-12">
        <div className="grid w-full grid-cols-1 gap-8 lg:grid-cols-2 lg:gap-16">
          {/* Left side - Status information */}
          <div className="flex flex-col justify-center space-y-8">
            {/* Animated Icon */}
            <div className="relative inline-flex h-32 w-32 items-center justify-center">
              {/* Pulsing rings */}
              <div className="absolute inset-0 animate-ping rounded-full bg-violet-500/20" />
              <div className="absolute inset-2 animate-pulse rounded-full bg-violet-500/30" />
              
              {/* Icon container */}
              <div className="relative inline-flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-[0_20px_50px_rgba(139,92,246,0.5)]">
                {statusInfo.icon}
              </div>
            </div>

            {/* Status text */}
            <div className="space-y-3">
              <h1 className="momo-trust-display-regular text-5xl text-white lg:text-6xl">
                {statusInfo.title}
              </h1>
              <p className="text-xl text-slate-400 lg:text-2xl">
                {message || statusInfo.subtitle}
              </p>
            </div>

            {/* Progress bar */}
            {displayProgress !== undefined && displayProgress > 0 && status !== "error" && (
              <div className="space-y-3">
                <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-500 ease-out"
                    style={{ width: `${displayProgress}%` }}
                  />
                </div>
                <p className="text-lg text-slate-500">{Math.round(displayProgress)}%</p>
              </div>
            )}

            {/* Error Actions */}
            {status === "error" && onRestart && (
              <div className="space-y-4 rounded-2xl border border-rose-500/20 bg-rose-500/10 p-6">
                <p className="text-sm text-rose-300">
                  The backend encountered an error while processing your video. Would you like to start a new session?
                </p>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={handleRestart}
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-6 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60"
                  >
                    Start New Session
                  </button>
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-6 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60"
                  >
                    Return to Login
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Right side - Scrolling fashion images */}
          <div className="relative hidden h-screen lg:flex">
            <div className="flex gap-4">
              {/* Column 1 - Scrolls up */}
              <div className="w-48 overflow-hidden">
                <div className="animate-scroll-up space-y-4">
                  {[...FASHION_IMAGES, ...FASHION_IMAGES].map((img, idx) => (
                    <div
                      key={`col1-${idx}`}
                      className="h-72 overflow-hidden rounded-2xl border border-white/10 shadow-lg"
                    >
                      <img
                        src={img}
                        alt="Fashion item"
                        className="h-full w-full object-cover"
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Column 2 - Scrolls down */}
              <div className="w-48 overflow-hidden">
                <div className="animate-scroll-down space-y-4">
                  {[...FASHION_IMAGES.slice(5), ...FASHION_IMAGES.slice(0, 5), ...FASHION_IMAGES.slice(5), ...FASHION_IMAGES.slice(0, 5)].map((img, idx) => (
                    <div
                      key={`col2-${idx}`}
                      className="h-72 overflow-hidden rounded-2xl border border-white/10 shadow-lg"
                    >
                      <img
                        src={img}
                        alt="Fashion item"
                        className="h-full w-full object-cover"
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Column 3 - Scrolls up */}
              <div className="w-48 overflow-hidden">
                <div className="animate-scroll-up space-y-4">
                  {[...FASHION_IMAGES.slice(10), ...FASHION_IMAGES.slice(0, 10), ...FASHION_IMAGES.slice(10), ...FASHION_IMAGES.slice(0, 10)].map((img, idx) => (
                    <div
                      key={`col3-${idx}`}
                      className="h-72 overflow-hidden rounded-2xl border border-white/10 shadow-lg"
                    >
                      <img
                        src={img}
                        alt="Fashion item"
                        className="h-full w-full object-cover"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
};

export default ProcessingScreen;
