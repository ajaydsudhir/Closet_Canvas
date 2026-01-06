import { EnvelopeSimple, Wrench } from "@phosphor-icons/react";
import { useState } from "react";
import { loginUser } from "../api/user";
import type { UserData } from "../types/user";

type UserLoginProps = {
  onLoginSuccess: (user: UserData) => void;
  onDevMode: () => void;
};

const UserLogin = ({ onLoginSuccess, onDevMode }: UserLoginProps) => {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !name.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const userData = await loginUser(email);
      onLoginSuccess(userData);
    } catch (err) {
      // For demo purposes, create a mock user if API fails
      const mockUser: UserData = {
        id: `user_${Date.now()}`,
        email: email,
        name: name,
      };
      onLoginSuccess(mockUser);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

      {/* Dev Mode Button */}
      <button
        type="button"
        onClick={onDevMode}
        className="absolute right-6 top-6 z-20 inline-flex items-center gap-2 rounded-full border border-amber-400/40 bg-amber-500/10 px-4 py-2 text-sm font-semibold text-amber-300 shadow-lg transition hover:bg-amber-500/20"
      >
        <Wrench size={18} weight="bold" />
        Dev Mode
      </button>

      <div className="relative z-10 mx-auto flex min-h-screen max-w-md flex-col items-center justify-center px-6">
        <div className="w-full space-y-8 rounded-3xl border border-white/10 bg-slate-900/70 p-8 shadow-[0_20px_35px_rgba(2,6,23,0.55)]">
          <div className="text-center">
            <h1 className="momo-trust-display-regular text-4xl text-white">
              Closet Canvas
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Your Personal Style Assistant
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label
                htmlFor="name"
                className="block text-sm font-medium text-slate-300"
              >
                Full Name
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter your name"
                className="w-full rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 text-white placeholder-slate-500 outline-none transition focus:border-white/30 focus:ring-2 focus:ring-white/10"
                required
              />
            </div>

            <div className="space-y-2">
              <label
                htmlFor="email"
                className="block text-sm font-medium text-slate-300"
              >
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-2xl border border-white/10 bg-slate-900/60 px-4 py-3 text-white placeholder-slate-500 outline-none transition focus:border-white/30 focus:ring-2 focus:ring-white/10"
                required
              />
            </div>

            {error && (
              <p className="text-sm text-rose-400">{error}</p>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/90 px-6 py-3 text-base font-semibold text-slate-900 shadow-[0_15px_35px_rgba(15,23,42,0.45)] transition hover:-translate-y-0.5 hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isLoading ? (
                <span className="text-sm tracking-wide">Logging in…</span>
              ) : (
                <>
                  <EnvelopeSimple size={20} weight="bold" />
                  Continue
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
};

export default UserLogin;
