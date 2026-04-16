import { useEffect, useState } from 'react';
import Head from 'next/head';

interface CameraStats {
  camera_id: string;
  entry_count: number;
  exit_count: number;
  current_crowd: number;
  unique_ids_count: number;
}

interface SummaryData {
  total_events: number;
  total_cameras: number;
  total_entries: number;
  total_exits: number;
  total_crowd: number;
  cameras: Record<string, CameraStats>;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const DEMO_DATA: SummaryData = {
  total_events: 1247,
  total_cameras: 2,
  total_entries: 542,
  total_exits: 528,
  total_crowd: 14,
  cameras: {
    cam_0: {
      camera_id: 'cam_0',
      entry_count: 320,
      exit_count: 312,
      current_crowd: 8,
      unique_ids_count: 156,
    },
    cam_1: {
      camera_id: 'cam_1',
      entry_count: 222,
      exit_count: 216,
      current_crowd: 6,
      unique_ids_count: 98,
    },
  },
};

export default function Dashboard() {
  const [summary, setSummary] = useState<SummaryData | null>(DEMO_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [useDemo, setUseDemo] = useState(false);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(`${API_URL}/summary`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setSummary(data);
        setError(null);
        setUseDemo(false);
      } catch (err) {
        console.error('API Error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setSummary(DEMO_DATA);
        setUseDemo(true);
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
    const interval = setInterval(fetchSummary, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <Head>
        <title>Event Tracking Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-4xl font-bold">📊 Event Tracking Dashboard</h1>
            {useDemo && <div className="bg-yellow-600 px-4 py-2 rounded text-sm">⚠️ Demo Mode</div>}
          </div>

          {summary && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
                <div className="bg-blue-600 p-6 rounded-lg shadow-lg">
                  <h3 className="text-gray-200 mb-2">Total Events</h3>
                  <p className="text-4xl font-bold">{summary.total_events}</p>
                </div>
                <div className="bg-purple-600 p-6 rounded-lg shadow-lg">
                  <h3 className="text-gray-200 mb-2">Cameras</h3>
                  <p className="text-4xl font-bold">{summary.total_cameras}</p>
                </div>
                <div className="bg-green-600 p-6 rounded-lg shadow-lg">
                  <h3 className="text-gray-200 mb-2">Entries</h3>
                  <p className="text-4xl font-bold">{summary.total_entries}</p>
                </div>
                <div className="bg-red-600 p-6 rounded-lg shadow-lg">
                  <h3 className="text-gray-200 mb-2">Exits</h3>
                  <p className="text-4xl font-bold">{summary.total_exits}</p>
                </div>
                <div className="bg-yellow-600 p-6 rounded-lg shadow-lg">
                  <h3 className="text-gray-200 mb-2">Current Crowd</h3>
                  <p className="text-4xl font-bold">{summary.total_crowd}</p>
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
                <h2 className="text-2xl font-bold mb-4">📹 Cameras</h2>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-700">
                        <th className="text-left p-3">Camera ID</th>
                        <th className="text-center p-3">Entries</th>
                        <th className="text-center p-3">Exits</th>
                        <th className="text-center p-3">Current Crowd</th>
                        <th className="text-center p-3">Unique IDs</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(summary.cameras).map(([_, camera]) => (
                        <tr key={camera.camera_id} className="border-b border-gray-700 hover:bg-gray-700">
                          <td className="p-3">{camera.camera_id}</td>
                          <td className="text-center p-3 text-green-400 font-semibold">{camera.entry_count}</td>
                          <td className="text-center p-3 text-red-400 font-semibold">{camera.exit_count}</td>
                          <td className="text-center p-3 text-yellow-400 font-semibold">{camera.current_crowd}</td>
                          <td className="text-center p-3 text-blue-400 font-semibold">{camera.unique_ids_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
