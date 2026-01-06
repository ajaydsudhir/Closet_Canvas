import { CoatHangerIcon } from "@phosphor-icons/react";

type LandingHeroProps = {
  isCreating: boolean;
  statusMessage: string | null;
  onCreateSession: () => void;
};

const LandingHero = ({
  isCreating,
  statusMessage,
  onCreateSession,
}: LandingHeroProps) => (
  <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
    <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

    <div className="relative z-10 mx-auto flex min-h-screen max-w-2xl flex-col items-center justify-center px-6 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-300">
        Capture Studio
      </p>

      <h1 className="momo-trust-display-regular mt-6 text-4xl text-white sm:text-5xl lg:text-6xl">
        Closet Canvas
      </h1>

      <button
        type="button"
        onClick={onCreateSession}
        disabled={isCreating}
        className="mt-10 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/90 px-6 py-3 text-base font-semibold text-slate-900 shadow-[0_15px_35px_rgba(15,23,42,0.45)] transition hover:-translate-y-0.5 hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isCreating ? (
          <span className="text-sm tracking-wide">Creating…</span>
        ) : (
          <>
            <span className="text-xl leading-none">
              <CoatHangerIcon size={24} />
            </span>
            Create Session
          </>
        )}
      </button>

      {statusMessage ? (
        <p className="mt-4 text-sm text-slate-300">{statusMessage}</p>
      ) : null}
    </div>
  </main>
);

export default LandingHero;
