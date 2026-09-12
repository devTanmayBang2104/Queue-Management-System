import React from 'react';

export default function VideoFeed({ frame }) {
  if (!frame) {
    return (
      <div className="video-placeholder">
        <div className="video-placeholder-icon">📹</div>
        <p style={{ fontSize: '1rem', fontWeight: 500 }}>No Video Feed</p>
        <p style={{ fontSize: '0.85rem' }}>
          Click <strong>Start</strong> to begin processing
        </p>
      </div>
    );
  }

  return (
    <img
      src={`data:image/jpeg;base64,${frame}`}
      alt="Live Queue Feed"
      style={{ background: '#000' }}
    />
  );
}
