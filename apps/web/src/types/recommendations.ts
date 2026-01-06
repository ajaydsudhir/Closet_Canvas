export type RecommendationItem = {
  id: string;
  imageUrl: string;
  title: string;
  brand?: string;
  price?: number;
  category?: string;
  matchScore?: number;
  fitScore?: number;
  preferenceScore?: number;
  size?: string;
};

export type RecommendationsResponse = {
  items: RecommendationItem[];
  sessionId: string;
};
