import React from 'react';

export default function ConfigPanel({ config, onUpdate }) {
  return (
    <div className="glass-card config-panel">
      <h3>⚙️ Alert Thresholds</h3>

      <div className="config-group">
        <div className="config-label">
          <span>Max Queue Length</span>
          <span className="config-value">{config.maxQueueLength}</span>
        </div>
        <input
          type="range"
          className="config-slider"
          min="3"
          max="30"
          value={config.maxQueueLength}
          onChange={(e) => onUpdate('maxQueueLength', parseInt(e.target.value))}
        />
      </div>

      <div className="config-group">
        <div className="config-label">
          <span>Max Wait Time (seconds)</span>
          <span className="config-value">{config.maxWaitTime}s</span>
        </div>
        <input
          type="range"
          className="config-slider"
          min="30"
          max="600"
          step="10"
          value={config.maxWaitTime}
          onChange={(e) => onUpdate('maxWaitTime', parseInt(e.target.value))}
        />
      </div>

      <div className="config-group">
        <div className="config-label">
          <span>Crowd Spike Threshold</span>
          <span className="config-value">{config.crowdThreshold}</span>
        </div>
        <input
          type="range"
          className="config-slider"
          min="5"
          max="50"
          value={config.crowdThreshold}
          onChange={(e) => onUpdate('crowdThreshold', parseInt(e.target.value))}
        />
      </div>
    </div>
  );
}
