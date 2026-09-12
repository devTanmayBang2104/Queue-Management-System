import React from 'react';

export default function HeatmapView({ heatmap }) {
  if (!heatmap) {
    return (
      <div className="heatmap-container" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        minHeight: '220px', color: 'var(--text-muted)', flexDirection: 'column', gap: '8px'
      }}>
        <span style={{ fontSize: '32px', opacity: 0.3 }}>🗺️</span>
        <span style={{ fontSize: '0.85rem' }}>Heatmap will appear when pipeline is running</span>
      </div>
    );
  }

  return (
    <div className="heatmap-container" style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
      <img
        src={`data:image/jpeg;base64,${heatmap}`}
        alt="Density Heatmap"
        style={{ width: '100%', display: 'block' }}
      />
    </div>
  );
}
