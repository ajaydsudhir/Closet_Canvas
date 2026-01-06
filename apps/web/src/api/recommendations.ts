const API_ROOT = "/api";

import type { RecommendationsResponse } from "../types/recommendations";

export async function fetchRecommendations(
  sessionId: string
): Promise<RecommendationsResponse> {
  const response = await fetch(
    `${API_ROOT}/v1/sessions/${sessionId}/recommendations`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch recommendations: ${response.status}`);
  }

  return response.json();
}

export async function submitRating(
  sessionId: string,
  garmentId: string,
  rating: number,
  userId?: string
): Promise<void> {
  const response = await fetch(
    `${API_ROOT}/v1/sessions/${sessionId}/ratings`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        garment_id: garmentId,
        rating,
        user_id: userId,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to submit rating: ${response.status}`);
  }
}
