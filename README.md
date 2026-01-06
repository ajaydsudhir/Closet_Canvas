# Closet Canvas: Fit and Style-Aware Recommendations for Apparel

## Summary
Closet Canvas matches users to clothes that fit and suit their style. We combine a user body profile with per-size garment specifications and multimodal style embeddings. The system retrieves style-relevant candidates, filters and sizes them via a calibrated fit model, then ranks with a multi-objective ranker. V1 excludes pricing and inventory to focus on the machine-learning core. Success is measured by higher size-hit rate, lower return-for-fit, and higher keep-rate within strict latency and fairness guardrails.

People know what they like, and models can learn from usage data. Most systems remain behavior-centric and ignore higher-order signals: stable style preferences and pre-computable priors about what a person will like. Closet Canvas fuses a user's body profile with fashion knowledge from the open web and partner catalogues to recommend items that both fit and match style.

Match users to clothes that fit their body and suit their style preferences.

Presentation Link: https://www.figma.com/proto/hr4RFKhDWlFpW0ia47LuRy/cc?node-id=151-274&p=f&t=Ntu76shCrqfmc0JP-0&scaling=scale-down&content-scaling=fixed&page-id=0%3A1&starting-point-node-id=47%3A659

## Run locally with Docker Compose
- Prereqs: Docker with Compose, and ports 3000, 8000, 5432, 6379, 9000, and 9001 available.
- From the repo root, copy the default envs: `cp .env.example .env` (set `HF_ACCESS_TOKEN` if you want the GPU worker to download models).
- Start the stack: `docker compose up --build` (first run may take a bit to fetch images/models).
- Open the frontend at http://localhost:3000 once containers finish starting.
