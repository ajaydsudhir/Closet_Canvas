import { useState } from "react";
import LandingHero from "./components/LandingHero";
import ProcessingScreen from "./components/ProcessingScreen";
import RecommendationsPage from "./components/RecommendationsPage";
import SessionWorkspace from "./components/SessionWorkspace";
import UserLogin from "./components/UserLogin";
import UserSessionWorkspace from "./components/UserSessionWorkspace";
import { createSession } from "./api/capture";
import { useSessionStatus } from "./hooks/useSessionStatus";
import type { SessionStatus } from "./types/session";
import type { UserData } from "./types/user";

type AppFlowState = 
  | "login"
  | "video-capture"
  | "processing"
  | "recommendations"
  | "dev-legacy"
  | "dev-processing"
  | "dev-recommendations";

const App = () => {
  // App flow state
  const [flowState, setFlowState] = useState<AppFlowState>(() => {
    // Check if URL has ?demo=processing or ?demo=recommendations for quick preview
    const params = new URLSearchParams(window.location.search);
    if (params.get('demo') === 'processing') {
      return 'dev-processing';
    }
    if (params.get('demo') === 'recommendations') {
      return 'dev-recommendations';
    }
    return 'login';
  });
  
  // User data
  const [user, setUser] = useState<UserData | null>(null);
  
  // Session state for video capture
  const [isCreating, setIsCreating] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [clipsReady, setClipsReady] = useState(false);

  // WebSocket status tracking (only enabled when in processing/recording state)
  const { status, message, progress } = useSessionStatus({
    sessionId: sessionId || "",
    enabled: flowState === "video-capture" || flowState === "processing",
    onStatusChange: handleStatusChange,
  });

  // Handler: Session status changes from WebSocket
  function handleStatusChange(newStatus: SessionStatus) {
    console.log("Session status changed:", newStatus);
    
    // Define status progression order (parallel processes may send out of order)
    const statusOrder: Record<SessionStatus, number> = {
      "idle": 0,
      "recording": 1,
      "gating": 2,
      "smpl": 3,
      "finishing": 4,
      "recommending": 5,
      "complete": 6,
      "error": -1, // Special case - always show errors
    };
    
    // Only transition to processing/complete if it's a progression (or error)
    const currentOrder = statusOrder[status || "idle"] || 0;
    const newOrder = statusOrder[newStatus] || 0;
    
    // Handle errors immediately
    if (newStatus === "error") {
      console.error("Session encountered an error");
      // Stay on current screen to show error
      return;
    }
    
    // Only update if new status is further along (ignore out-of-order updates)
    if (newOrder < currentOrder) {
      console.log(`Ignoring out-of-order status: ${newStatus} (current: ${status})`);
      return;
    }
    
    // Automatically transition to processing screen when backend starts processing
    // BUT only if we've sent at least 5 clips
    if (newStatus === "recording" || newStatus === "gating" || newStatus === "smpl" || newStatus === "finishing" || newStatus === "recommending") {
      if (clipsReady) {
        setFlowState("processing");
      }
    } 
    // Transition to recommendations when complete
    else if (newStatus === "complete") {
      setFlowState("recommendations");
    }
  }

  // Handler: User login complete - create session immediately
  const handleLoginSuccess = async (userData: UserData) => {
    setUser(userData);
    await handleCreateSession();
    setFlowState("video-capture");
  };

  // Handler: Create session for video capture
  const handleCreateSession = async () => {
    if (isCreating) return;

    setIsCreating(true);
    setStatusMessage(null);

    try {
      const payload = await createSession();
      if (!payload.session_id) {
        throw new Error("Session created but response was missing an id");
      }
      setSessionId(payload.session_id);
      setStatusMessage(`Session created: ${payload.session_id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(`Unable to create session: ${message}`);
      // For demo, create a mock session ID
      setSessionId(`session_${Date.now()}`);
    } finally {
      setIsCreating(false);
    }
  };

  // Handler: Dev mode access (go to legacy landing page)
  const handleDevMode = () => {
    setFlowState("dev-legacy");
  };

  // Handler: Create session from dev mode
  const handleDevCreateSession = async () => {
    if (isCreating) return;

    setIsCreating(true);
    setStatusMessage(null);

    try {
      const payload = await createSession();
      if (!payload.session_id) {
        throw new Error("Session created but response was missing an id");
      }
      setSessionId(payload.session_id);
      setStatusMessage(`Session created: ${payload.session_id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(`Unable to create session: ${message}`);
    } finally {
      setIsCreating(false);
    }
  };

  // Render based on flow state
  switch (flowState) {
    case "login":
      return <UserLogin onLoginSuccess={handleLoginSuccess} onDevMode={handleDevMode} />;

    case "video-capture":
      if (!user || !sessionId) {
        return (
          <div className="flex min-h-screen items-center justify-center bg-slate-950 text-white">
            <p>Initializing session...</p>
          </div>
        );
      }
      return (
        <UserSessionWorkspace
          sessionId={sessionId}
          user={user}
          onClipsReady={() => {
            console.log("5 clips ready callback - transitioning to processing");
            setClipsReady(true);
            setFlowState("processing");
          }}
          onCameraError={() => setFlowState("login")}
        />
      );

    case "processing":
      return (
        <ProcessingScreen 
          status={status || "idle"} 
          message={message} 
          progress={progress} 
          onLogout={() => setFlowState("login")}
          onRestart={async () => {
            setSessionId(null);
            setClipsReady(false);
            await handleCreateSession();
            setFlowState("video-capture");
          }}
        />
      );

    case "dev-processing":
      // Demo view with cycling statuses
      return <ProcessingScreen status="smpl" message="Demo: Analyzing your body measurements..." progress={67} onLogout={() => window.location.href = '/'} />;

    case "dev-recommendations":
      // Demo view with mock user and session
      return (
        <RecommendationsPage 
          user={{ id: "demo_user", name: "Demo User", email: "demo@example.com" }} 
          sessionId="demo_session_123"
          onLogout={() => window.location.href = '/'}
          onRestart={() => window.location.href = '/'}
        />
      );

    case "recommendations":
      if (!user || !sessionId) return null;
      return (
        <RecommendationsPage 
          user={user} 
          sessionId={sessionId}
          onLogout={() => setFlowState("login")}
          onRestart={async () => {
            setSessionId(null);
            setClipsReady(false);
            await handleCreateSession();
            setFlowState("video-capture");
          }}
        />
      );

    case "dev-legacy":
      if (sessionId) {
        return <SessionWorkspace sessionId={sessionId} />;
      }
      return (
        <LandingHero
          isCreating={isCreating}
          statusMessage={statusMessage}
          onCreateSession={handleDevCreateSession}
        />
      );

    default:
      return null;
  }
};

export default App;
