import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 300 },
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(12, 16, 32, 0.9)',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      titleFont: { family: 'Inter', weight: '600' },
      bodyFont: { family: 'Inter' },
      padding: 10,
      cornerRadius: 8,
    },
  },
  scales: {
    x: {
      display: true,
      grid: { display: false },
      ticks: { maxTicksLimit: 8, font: { size: 10 } },
    },
    y: {
      display: true,
      beginAtZero: true,
      grid: { color: 'rgba(255,255,255,0.04)' },
      ticks: { font: { size: 10 } },
    },
  },
  elements: {
    point: { radius: 0, hoverRadius: 4 },
    line: { tension: 0.4, borderWidth: 2 },
  },
};

export default function Charts({ history }) {
  const labels = useMemo(() => {
    return history.map((h, i) => {
      if (i % 10 === 0) {
        const d = new Date(h.timestamp * 1000);
        return d.toLocaleTimeString([], { minute: '2-digit', second: '2-digit' });
      }
      return '';
    });
  }, [history]);

  const queueData = useMemo(() => ({
    labels,
    datasets: [{
      label: 'Queue Length',
      data: history.map(h => h.queue_count),
      borderColor: '#00d4ff',
      backgroundColor: 'rgba(0, 212, 255, 0.1)',
      fill: true,
    }],
  }), [history, labels]);

  const waitData = useMemo(() => ({
    labels,
    datasets: [
      {
        label: 'Avg Wait',
        data: history.map(h => h.avg_wait_time),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
      },
      {
        label: 'Predicted',
        data: history.map(h => h.predicted_wait_time),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.05)',
        fill: false,
        borderDash: [5, 5],
      },
    ],
  }), [history, labels]);

  const waitOptions = {
    ...commonOptions,
    plugins: {
      ...commonOptions.plugins,
      legend: {
        display: true,
        position: 'top',
        align: 'end',
        labels: {
          boxWidth: 12,
          boxHeight: 2,
          font: { size: 11, family: 'Inter' },
          padding: 16,
        },
      },
    },
  };

  return (
    <>
      <div className="glass-card chart-card">
        <h3>📊 Queue Length Over Time</h3>
        <div className="chart-wrapper">
          <Line data={queueData} options={commonOptions} />
        </div>
      </div>

      <div className="glass-card chart-card">
        <h3>⏱️ Wait Time Trends</h3>
        <div className="chart-wrapper">
          <Line data={waitData} options={waitOptions} />
        </div>
      </div>
    </>
  );
}
