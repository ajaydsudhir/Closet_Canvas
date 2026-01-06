const API_ROOT = "/api";

import type { UserData, UserMeasurements } from "../types/user";

export async function loginUser(email: string): Promise<UserData> {
  const response = await fetch(`${API_ROOT}/v1/users/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });

  if (!response.ok) {
    throw new Error(`Failed to login: ${response.status}`);
  }

  return response.json();
}

export async function submitMeasurements(
  userId: string,
  measurements: UserMeasurements
): Promise<void> {
  const response = await fetch(`${API_ROOT}/v1/users/${userId}/measurements`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(measurements),
  });

  if (!response.ok) {
    throw new Error(`Failed to submit measurements: ${response.status}`);
  }
}
