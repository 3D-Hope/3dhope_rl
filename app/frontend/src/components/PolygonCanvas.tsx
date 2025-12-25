import { useEffect, useRef } from 'react';
import type { PolygonPoint } from '../types';

interface PolygonCanvasProps {
  polygonPoints: PolygonPoint[];
  onCanvasClick: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  onClear: () => void;
  disabled: boolean;
}

export function PolygonCanvas({
  polygonPoints,
  onCanvasClick,
  onClear,
  disabled,
}: PolygonCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const pos = (i / 6) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(canvas.width, pos);
      ctx.stroke();
    }

    // Draw polygon points and lines
    if (polygonPoints.length > 0) {
      ctx.fillStyle = '#2196F3';
      ctx.strokeStyle = '#2196F3';
      ctx.lineWidth = 2;

      // Convert world coordinates to canvas coordinates
      const canvasPoints = polygonPoints.map((p) => ({
        x: ((p.x + 3) / 6) * canvas.width,
        z: ((p.z + 3) / 6) * canvas.height,
      }));

      // Draw lines
      ctx.beginPath();
      ctx.moveTo(canvasPoints[0].x, canvasPoints[0].z);
      for (let i = 1; i < canvasPoints.length; i++) {
        ctx.lineTo(canvasPoints[i].x, canvasPoints[i].z);
      }
      if (polygonPoints.length >= 3) {
        ctx.closePath();
      }
      ctx.stroke();

      // Draw points
      canvasPoints.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.z, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText((i + 1).toString(), p.x, p.z - 10);
        ctx.fillStyle = '#2196F3';
      });
    }
  }, [polygonPoints]);

  return (
    <div style={{ border: '2px solid #ddd', borderRadius: '4px', padding: '10px', marginBottom: '10px' }}>
      <canvas
        ref={canvasRef}
        width={400}
        height={400}
        onClick={onCanvasClick}
        style={{
          cursor: disabled ? 'not-allowed' : 'crosshair',
          border: '1px solid #ccc',
          display: 'block',
          margin: '0 auto',
          opacity: disabled ? 0.6 : 1,
        }}
      />
      <div style={{ marginTop: '10px', textAlign: 'center' }}>
        <button
          type="button"
          onClick={onClear}
          disabled={disabled || polygonPoints.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: disabled || polygonPoints.length === 0 ? 'not-allowed' : 'pointer',
            opacity: disabled || polygonPoints.length === 0 ? 0.5 : 1,
          }}
        >
          Clear Polygon
        </button>
        <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
          Click on the grid to add points. Need at least 3 points to form a polygon.
          <br />
          Points: {polygonPoints.length}
        </p>
        {polygonPoints.length > 0 && (
          <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
            <strong>Current points:</strong>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {polygonPoints.map((p, i) => (
                <li key={i}>
                  Point {i + 1}: x={p.x.toFixed(2)}, z={p.z.toFixed(2)}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
