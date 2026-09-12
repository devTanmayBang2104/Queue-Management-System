import React from 'react';

export default function AlertPanel({ alerts }) {
  const formatTime = (ts) => {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
  };

  const severityIcon = {
    critical: '🔴',
    warning: '🟡',
    info: '🔵',
  };

  return (
    <div className="glass-card alert-panel">
      <h3>
        <span>🔔</span> Alerts & Notifications
        {alerts.length > 0 && (
          <span style={{
            background: 'var(--accent-rose)',
            color: 'white',
            padding: '2px 8px',
            borderRadius: '100px',
            fontSize: '0.7rem',
            fontWeight: 700,
            marginLeft: '8px',
          }}>
            {alerts.length}
          </span>
        )}
      </h3>

      {alerts.length === 0 ? (
        <div className="no-alerts">
          <p>✅ No active alerts</p>
          <p style={{ marginTop: '4px', fontSize: '0.8rem' }}>
            Alerts trigger when thresholds are exceeded
          </p>
        </div>
      ) : (
        alerts.slice(0, 20).map((alert, index) => (
          <div
            key={`${alert.timestamp}-${index}`}
            className={`alert-item ${alert.severity || 'info'}`}
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            <div className="alert-message">
              {severityIcon[alert.severity] || '🔵'} {alert.message}
            </div>
            <div className="alert-time">
              {formatTime(alert.timestamp)} · {alert.type?.replace('_', ' ')}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
