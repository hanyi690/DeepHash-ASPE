'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DemoPage() {
  const [queryType, setQueryType] = useState<'text' | 'image'>('text');
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [searchTime, setSearchTime] = useState(0);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (queryType === 'text' && !queryText.trim()) {
      setError('请输入查询文本');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE}/api/search`, {
        query_type: queryType,
        query_text: queryType === 'text' ? queryText : undefined,
        top_k: 10,
        use_encrypted: true
      });

      setResults(response.data.results || []);
      setSearchTime(response.data.search_time_ms || 0);
    } catch (err: any) {
      setError(err.response?.data?.detail || '搜索失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const sampleQueries = [
    "A cat sitting on a chair",
    "A dog playing in the park",
    "A beautiful sunset over the ocean",
    "People walking on the street"
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">跨模态检索演示</h1>
        <p className="text-gray-600">
          输入文本或上传图像，系统将返回最相关的图像结果
        </p>
      </div>

      {/* Query Type Selector */}
      <div className="card mb-6">
        <div className="flex space-x-4 mb-4">
          <button
            onClick={() => setQueryType('text')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              queryType === 'text'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            文本查询
          </button>
          <button
            onClick={() => setQueryType('image')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              queryType === 'image'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            图像查询
          </button>
        </div>

        {/* Query Input */}
        {queryType === 'text' ? (
          <div>
            <textarea
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="输入描述性文本，例如：A cat sitting on a chair"
              className="input-primary mb-4 h-24 resize-none"
            />
            <div className="flex flex-wrap gap-2 mb-4">
              <span className="text-sm text-gray-500">示例查询:</span>
              {sampleQueries.map((q, i) => (
                <button
                  key={i}
                  onClick={() => setQueryText(q)}
                  className="text-xs bg-gray-100 hover:bg-gray-200 px-2 py-1 rounded text-gray-700"
                >
                  {q.length > 30 ? q.substring(0, 30) + '...' : q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <div className="text-gray-400 mb-2">📷</div>
            <p className="text-gray-600 mb-4">点击或拖拽上传图像</p>
            <button className="btn-primary">选择图像</button>
            <p className="text-sm text-gray-500 mt-2">
              （图像上传功能开发中）
            </p>
          </div>
        )}

        <button
          onClick={handleSearch}
          disabled={loading}
          className="btn-primary w-full mt-4"
        >
          {loading ? '搜索中...' : '开始检索'}
        </button>
      </div>

      {/* Search Info */}
      {results.length > 0 && (
        <div className="mb-4 text-sm text-gray-600">
          检索完成，耗时 {searchTime.toFixed(2)} ms，找到 {results.length} 个结果
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {results.map((result, index) => (
            <div key={index} className="card">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-500">
                  #{result.rank} - 图像 {result.image_id}
                </span>
                <span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded">
                  距离：{result.distance.toFixed(2)}
                </span>
              </div>
              <div className="bg-gray-100 rounded-lg h-40 mb-3 flex items-center justify-center">
                <span className="text-gray-400">图像预览</span>
              </div>
              {result.captions && result.captions.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 mb-1">相关描述:</p>
                  <ul className="text-sm text-gray-700 space-y-1">
                    {result.captions.map((caption: string, i: number) => (
                      <li key={i} className="flex items-start">
                        <span className="text-gray-400 mr-2">•</span>
                        <span>{caption}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
