import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler,
} from 'chart.js';
import VideoFeed from './components/VideoFeed';
import QueueStats from './components/QueueStats';
import Charts from './components/Charts';
import HeatmapView from './components/HeatmapView';
import AlertPanel from './components/AlertPanel';
import ConfigPanel from './components/ConfigPanel';

// Register Chart.js components
ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
);

// Chart.js global defaults for dark theme
ChartJS.defaults.color = '#8892a8';
ChartJS.defaults.borderColor = 'rgba(255,255,255,0.06)';
ChartJS.defaults.font.family = 'Inter';

const WS_URL = `ws://${window.location.hostname}:8000/ws`;
const API_BASE = `http://${window.location.hostname}:8000`;

export default function App() {
  // State
  const [connected, setConnected] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [frame, setFrame] = useState(null);
  const [stats, setStats] = useState({});
  const [persons, setPersons] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [heatmap, setHeatmap] = useState(null);
  const [history, setHistory] = useState([]);
  const [videoSource, setVideoSource] = useState('');

  // Config state
  const [config, setConfig] = useState({
    maxQueueLength: 10,
    maxWaitTime: 300,
    crowdThreshold: 15,
  });

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  // ── WebSocket Connection ──────────────────────────────

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
      console.log('[WS] Connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'frame_data') {
          setFrame(data.frame);
          setStats(data.stats || {});
          setPersons(data.persons || []);

          // Accumulate alerts (deduplicate)
          if (data.alerts?.length > 0) {
            setAlerts(prev => {
              const newAlerts = [...data.alerts, ...prev].slice(0, 50);
              return newAlerts;
            });
          }

          // Add to history for charts
          if (data.stats) {
            setHistory(prev => {
              const point = {
                timestamp: data.stats.timestamp,
                queue_count: data.stats.queue_count,
                avg_wait_time: data.stats.avg_wait_time,
                predicted_wait_time: data.stats.predicted_wait_time,
              };
              const updated = [...prev, point].slice(-120); // Keep last 120 points
              return updated;
            });
          }
        }
      } catch (e) {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('[WS] Disconnected');
      // Auto-reconnect after 3s
      reconnectTimer.current = setTimeout(connectWS, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connectWS]);

  // Periodically fetch heatmap
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!pipelineRunning) return;
      try {
        const res = await fetch(`${API_BASE}/api/heatmap`);
        const data = await res.json();
        if (data.heatmap) setHeatmap(data.heatmap);
      } catch (e) { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, [pipelineRunning]);

  // ── Pipeline Control ──────────────────────────────────

  const startPipeline = async () => {
    try {
      const source = videoSource || '0';
      const res = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_source: source }),
      });
      const data = await res.json();
      if (data.status === 'started' || data.status === 'already_running') {
        setPipelineRunning(true);
      }
    } catch (e) {
      console.error('Failed to start pipeline:', e);
    }
  };

  const stopPipeline = async () => {
    try {
      await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
      setPipelineRunning(false);
      setFrame(null);
    } catch (e) {
      console.error('Failed to stop pipeline:', e);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.path) {
        setVideoSource(data.path);
      }
    } catch (e) {
      console.error('Upload failed:', e);
    }
  };

  const updateConfig = async (key, value) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    try {
      await fetch(`${API_BASE}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_queue_length_alert: newConfig.maxQueueLength,
          max_wait_time_alert: newConfig.maxWaitTime,
          crowd_spike_threshold: newConfig.crowdThreshold,
        }),
      });
    } catch (e) { /* ignore */ }
  };

  // ── Render ────────────────────────────────────────────

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">👁️</div>
          <div>
            <h1>QueueVision AI</h1>
            <span>Intelligent Queue Analytics</span>
          </div>
        </div>

        <div className="header-status">
          <div className="controls-bar">
            <input
              type="text"
              className="source-input"
              placeholder="Video source (0 for webcam, or file path)"
              value={videoSource}
              onChange={(e) => setVideoSource(e.target.value)}
            />
            <label className="btn btn-outline" style={{ cursor: 'pointer' }}>
              📁 Upload
              <input type="file" accept="video/*" onChange={handleUpload}
                     style={{ display: 'none' }} />
            </label>
            {!pipelineRunning ? (
              <button className="btn btn-primary" onClick={startPipeline}>
                ▶ Start
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopPipeline}>
                ⏹ Stop
              </button>
            )}
          </div>

          <div className={`status-badge ${connected ? 'online' : 'offline'}`}>
            <span className="status-dot"></span>
            {connected ? 'Connected' : 'Disconnected'}
          </div>

          {stats.fps > 0 && (
            <div className="fps-badge">{stats.fps} FPS</div>
          )}
        </div>
      </header>

      {/* Main Dashboard Grid */}
      <main className="main-content">
        {/* Stats Row */}
        <QueueStats stats={stats} />

        {/* Video Feed */}
        <div className="glass-card video-container">
          <VideoFeed frame={frame} />
        </div>

        {/* Charts Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <Charts history={history} />
          <ConfigPanel config={config} onUpdate={updateConfig} />
        </div>

        {/* Heatmap */}
        <div className="glass-card" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '12px', fontWeight: 600 }}>
            🔥 Density Heatmap
          </h3>
          <HeatmapView heatmap={heatmap} />
        </div>

        {/* Alerts */}
        <AlertPanel alerts={alerts} />
      </main>
    </div>
  );
}
