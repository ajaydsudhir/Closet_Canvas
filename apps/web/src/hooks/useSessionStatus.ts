import { useEffect, useRef, useState } from "react";
import type { SessionStatus, WebSocketMessage } from "../types/session";

type UseSessionStatusOptions = {
  sessionId: string;
  enabled?: boolean;
  onStatusChange?: (status: SessionStatus) => void;
};

export const useSessionStatus = ({
  sessionId,
  enabled = true,
  onStatusChange,
}: UseSessionStatusOptions) => {
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [message, setMessage] = useState<string | undefined>();
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const onStatusChangeRef = useRef(onStatusChange);

  // Update callback ref when it changes
  useEffect(() => {
    onStatusChangeRef.current = onStatusChange;
  }, [onStatusChange]);

  // Status progression order to handle out-of-order updates
  const statusOrderRef = useRef<Record<SessionStatus, number>>({
    "idle": 0,
    "recording": 1,
    "gating": 2,
    "smpl": 3,
    "finishing": 4,
    "recommending": 5,
    "complete": 6,
    "error": -1, // Special case - always show errors
  });

  useEffect(() => {
    if (!enabled || !sessionId) return;

    const connect = () => {
      try {
        // Determine WebSocket URL based on current location
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/api/v1/sessions/${sessionId}/status`;
        
        console.log(`Attempting to connect WebSocket to: ${wsUrl}`);
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log(`WebSocket connected for session ${sessionId}`);
          setIsConnected(true);
          setError(null);
          reconnectAttempts.current = 0;
        };

        ws.onmessage = (event) => {
          try {
            const data: WebSocketMessage = JSON.parse(event.data);
            
            if (data.type === "status") {
              // Check if this is a progression (handle parallel processing out-of-order)
              setStatus((currentStatus) => {
                const currentOrder = statusOrderRef.current[currentStatus] || 0;
                const newOrder = statusOrderRef.current[data.status] || 0;
                
                // Always accept errors
                if (data.status === "error") {
                  setMessage(data.message);
                  setProgress(data.progress ?? 0);
                  setError(data.message || "An error occurred");
                  onStatusChangeRef.current?.(data.status);
                  return data.status;
                }
                
                // Only update if new status is further along (or same)
                if (newOrder >= currentOrder) {
                  setMessage(data.message);
                  setProgress(data.progress ?? 0);
                  onStatusChangeRef.current?.(data.status);
                  return data.status;
                } else {
                  console.log(`[WebSocket] Ignoring out-of-order status: ${data.status} (current: ${currentStatus})`);
                  return currentStatus;
                }
              });
            } else if (data.type === "error") {
              setError(data.error);
              setStatus("error");
            }
          } catch (err) {
            console.error("Failed to parse WebSocket message:", err);
          }
        };

        ws.onerror = (event) => {
          console.error("WebSocket error:", event);
          setError("Connection error");
        };

        ws.onclose = () => {
          console.log("WebSocket closed");
          setIsConnected(false);
          wsRef.current = null;

          // Attempt to reconnect
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, delay);
          } else {
            setError("Unable to connect to server. Please refresh the page.");
          }
        };
      } catch (err) {
        console.error("Failed to create WebSocket:", err);
        setError("Failed to establish connection");
      }
    };

    // Initial connection
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sessionId, enabled]); // Removed onStatusChange from dependencies

  return {
    status,
    message,
    progress,
    error,
    isConnected,
  };
};
