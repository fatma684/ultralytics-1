// pages/index.tsx - Dashboard Home Page
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

export default function Dashboard() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch('http://localhost:8000/summary');
        if (!response.ok) throw new Error('Failed to fetch summary');
        const data = await response.json();
        setSummary(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
    const interval = setInterval(fetchSummary, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div className="text-center p-8">Loading...</div>;
  if (error) return <div className="text-red-500 p-8">Error: {error}</div>;

  return (
    <>
      <Head>
        <title>Event Tracking Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-4xl font-bold mb-8">📊 Event Tracking Dashboard</h1>

          {summary && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
                <div className="bg-blue-600 p-6 rounded-lg">
                  <h3 className="text-gray-200 mb-2">Total Events</h3>
                  <p className="text-4xl font-bold">{summary.total_events}</p>
                </div>
                <div className="bg-purple-600 p-6 rounded-lg">
                  <h3 className="text-gray-200 mb-2">Cameras</h3>
                  <p className="text-4xl font-bold">{summary.total_cameras}</p>
                </div>
                <div className="bg-green-600 p-6 rounded-lg">
                  <h3 className="text-gray-200 mb-2">Entries</h3>
                  <p className="text-4xl font-bold">{summary.total_entries}</p>
                </div>
                <div className="bg-red-600 p-6 rounded-lg">
                  <h3 className="text-gray-200 mb-2">Exits</h3>
                  <p className="text-4xl font-bold">{summary.total_exits}</p>
                </div>
                <div className="bg-yellow-600 p-6 rounded-lg">
                  <h3 className="text-gray-200 mb-2">Current Crowd</h3>
                  <p className="text-4xl font-bold">{summary.total_crowd}</p>
                </div>
              </div>

              {/* Cameras Table */}
              <div className="bg-gray-800 rounded-lg p-6">
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
                          <td className="text-center p-3 text-green-400">{camera.entry_count}</td>
                          <td className="text-center p-3 text-red-400">{camera.exit_count}</td>
                          <td className="text-center p-3 text-yellow-400">{camera.current_crowd}</td>
                          <td className="text-center p-3 text-blue-400">{camera.unique_ids_count}</td>
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
