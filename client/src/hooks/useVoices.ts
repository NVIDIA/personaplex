import { useState, useEffect } from 'react';

export interface Voice {
  name: string;
  type: 'embeddings' | 'audio';
  category: 'custom' | 'natural-female' | 'natural-male' | 'variety-female' | 'variety-male' | 'other';
  path: string;
}

export interface UseVoicesReturn {
  voices: Voice[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useVoices(): UseVoicesReturn {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVoices = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/voices');
      if (!response.ok) {
        throw new Error(`Failed to fetch voices: ${response.statusText}`);
      }

      const data = await response.json();
      setVoices(data.voices || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching voices:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVoices();
  }, []);

  return {
    voices,
    loading,
    error,
    refresh: fetchVoices,
  };
}
