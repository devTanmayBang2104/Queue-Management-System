import React, { useEffect, useRef } from 'react';

function StatCard({ label, value, sub, accent }) {
  const prevValue = useRef(value);

  useEffect(() => {
    prevValue.current = value;
  }, [value]);

  return (
    <div className={`glass-card stat-card ${accent} animate-fade-in`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export default function QueueStats({ stats }) {
  const queueCount = stats.queue_count ?? 0;
  const avgWait = stats.avg_wait_time ?? 0;
  const maxWait = stats.max_wait_time ?? 0;
  const predicted = stats.predicted_wait_time ?? 0;
  const totalTracked = stats.total_tracked ?? 0;
  const activeTracked = stats.active_tracks ?? 0;
  const abandoned = stats.total_abandoned ?? 0;
  const served = stats.total_served ?? 0;

  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  };

  return (
    <div className="stats-row">
      <StatCard
        label="Queue Count"
        value={queueCount}
        sub={`${activeTracked} active · ${totalTracked} total tracked`}
        accent="cyan"
      />
      <StatCard
        label="Avg Wait Time"
        value={formatTime(avgWait)}
        sub={`Max: ${formatTime(maxWait)}`}
        accent="emerald"
      />
      <StatCard
        label="Predicted Wait"
        value={formatTime(predicted)}
        sub="For next person in queue"
        accent="amber"
      />
      <StatCard
        label="Served / Abandoned"
        value={`${served} / ${abandoned}`}
        sub={`Abandon rate: ${((stats.abandonment_rate ?? 0) * 100).toFixed(1)}%`}
        accent="purple"
      />
    </div>
  );
}
