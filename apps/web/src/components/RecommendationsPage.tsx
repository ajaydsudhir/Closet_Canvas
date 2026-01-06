import { ArrowClockwise, SignOut, Sparkle } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { fetchRecommendations } from "../api/recommendations";
import type { RecommendationItem } from "../types/recommendations";
import type { UserData } from "../types/user";

type RecommendationsPageProps = {
  user: UserData;
  sessionId: string;
  onLogout?: () => void;
  onRestart?: () => void;
};

const RecommendationsPage = ({ user, sessionId, onLogout, onRestart }: RecommendationsPageProps) => {
  const [items, setItems] = useState<RecommendationItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadRecommendations();
  }, [sessionId]);

  const loadRecommendations = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetchRecommendations(sessionId);
      setItems(response.items);
    } catch (err) {
      // For demo purposes, show mock data if API fails
      setItems(generateMockRecommendations());
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    } else {
      // Default: reload to login
      window.location.href = '/';
    }
  };

  const handleRestart = () => {
    if (onRestart) {
      onRestart();
    } else {
      // Default: reload to start fresh
      window.location.href = '/';
    }
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="bg-dots absolute inset-0 opacity-70" aria-hidden="true" />

      <div className="relative z-10 mx-auto max-w-7xl px-6 py-10">
        {/* Header */}
        <header className="mb-8 flex items-center justify-between rounded-3xl border border-white/10 bg-slate-900/70 px-6 py-4 shadow-[0_20px_35px_rgba(2,6,23,0.55)]">
          <div className="flex items-center gap-4">
            <div className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500">
              <Sparkle size={24} weight="fill" className="text-white" />
            </div>
            <div>
              <h1 className="momo-trust-display-regular text-2xl text-white">
                Your Recommendations
              </h1>
              <p className="text-sm text-slate-400">
                Curated just for you, {user.name}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleRestart}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60"
            >
              <ArrowClockwise size={18} weight="bold" />
              <span>New Session</span>
            </button>
            <button
              type="button"
              onClick={handleLogout}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-white/20 hover:bg-slate-800/60"
            >
              <SignOut size={18} weight="bold" />
              <span>Logout</span>
            </button>
          </div>
        </header>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="mb-4 inline-flex h-16 w-16 animate-spin items-center justify-center rounded-full border-4 border-white/10 border-t-white/80" />
              <p className="text-sm text-slate-400">
                Finding your perfect matches...
              </p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-6 py-4 text-center">
            <p className="text-sm text-rose-300">{error}</p>
          </div>
        )}

        {/* Masonry Grid - Pinterest Style */}
        {!isLoading && items.length > 0 && (
          <div className="masonry-grid">
            {items.map((item) => (
              <RecommendationCard
                key={item.id}
                item={item}
              />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!isLoading && items.length === 0 && !error && (
          <div className="rounded-2xl border border-white/10 bg-slate-900/40 px-6 py-20 text-center">
            <p className="text-slate-400">
              No recommendations available yet. Complete your video capture to
              get personalized suggestions!
            </p>
          </div>
        )}
      </div>
    </main>
  );
};

const RecommendationCard = ({
  item,
}: {
  item: RecommendationItem;
}) => (
  <div className="masonry-item group relative overflow-hidden rounded-2xl border border-white/10 bg-slate-900/60 shadow-lg transition hover:border-white/20 hover:shadow-2xl">
    <div className="relative overflow-hidden bg-slate-800">
      <img
        src={item.imageUrl}
        alt={item.title}
        className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
        loading="lazy"
      />
      {item.matchScore && (
        <div className="absolute bottom-3 left-3 inline-flex items-center gap-1 rounded-full border border-white/20 bg-slate-900/80 px-3 py-1 backdrop-blur-sm">
          <Sparkle size={14} weight="fill" className="text-amber-400" />
          <span className="text-xs font-semibold text-white">
            {Math.round(item.matchScore * 100)}% Match
          </span>
        </div>
      )}
      {item.size && (
        <div className="absolute top-3 right-3 inline-flex items-center rounded-full border border-white/20 bg-slate-900/80 px-2 py-0.5 backdrop-blur-sm">
          <span className="text-xs font-medium text-slate-200">
            Size {item.size}
          </span>
        </div>
      )}
    </div>
    <div className="p-4">
      <h3 className="text-sm font-semibold text-white truncate">{item.title}</h3>
      {item.brand && (
        <p className="text-xs text-slate-400 mt-1">{item.brand}</p>
      )}
      {item.price && (
        <p className="text-sm font-medium text-emerald-400 mt-2">${item.price.toFixed(2)}</p>
      )}
      {(item.fitScore !== undefined || item.preferenceScore !== undefined) && (
        <div className="mt-3 flex gap-2 text-xs">
          {item.fitScore !== undefined && (
            <span className="text-slate-400">
              Fit: <span className="text-slate-200">{Math.round(item.fitScore)}%</span>
            </span>
          )}
          {item.preferenceScore !== undefined && (
            <span className="text-slate-400">
              Style: <span className="text-slate-200">{Math.round(item.preferenceScore)}%</span>
            </span>
          )}
        </div>
      )}
    </div>
  </div>
);

// Mock data generator for demo purposes
const generateMockRecommendations = (): RecommendationItem[] => {
  const categories = ["Tops", "Bottoms", "Dresses", "Outerwear", "Accessories"];
  const brands = ["StyleCo", "UrbanThread", "ChicBoutique", "ModernWear"];
  
  // Array of random heights for Pinterest-style variety
  const heights = [400, 500, 600, 450, 550, 650, 420, 520, 580, 480, 530, 600];
  
  return Array.from({ length: 12 }, (_, i) => ({
    id: `item_${i + 1}`,
    imageUrl: `https://picsum.photos/seed/${i + 100}/400/${heights[i]}`,
    title: `Stylish Item ${i + 1}`,
    brand: brands[Math.floor(Math.random() * brands.length)],
    price: Math.floor(Math.random() * 150) + 30,
    category: categories[Math.floor(Math.random() * categories.length)],
    matchScore: 0.7 + Math.random() * 0.3,
  }));
};

export default RecommendationsPage;
