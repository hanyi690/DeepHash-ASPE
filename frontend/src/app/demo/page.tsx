'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import ResultsGrid from '@/components/ResultsGrid';

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
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">跨模态检索演示</h1>
        <p className="text-gray-600">
          输入文本或上传图像，系统将返回最相关的图像结果
        </p>
      </div>

      {/* 查询卡片 */}
      <div className="card mb-8">
        {/* 查询类型切换 */}
        <div className="flex space-x-2 mb-6 p-1 bg-gray-100 rounded-xl w-fit">
          <button
            onClick={() => setQueryType('text')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              queryType === 'text'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span>文本查询</span>
            </div>
          </button>
          <button
            onClick={() => setQueryType('image')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              queryType === 'image'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>图像查询</span>
            </div>
          </button>
        </div>

        {/* 查询输入 */}
        {queryType === 'text' ? (
          <div>
            <textarea
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              placeholder="输入描述性文本，例如：A cat sitting on a chair"
              className="input-primary mb-4 h-28 resize-none"
            />
            <div className="flex flex-wrap gap-2 mb-4">
              <span className="text-sm text-gray-500 self-center">示例查询:</span>
              {sampleQueries.map((q, i) => (
                <button
                  key={i}
                  onClick={() => setQueryText(q)}
                  className="badge-neutral hover:bg-gray-200 transition-colors cursor-pointer text-xs"
                >
                  {q.length > 25 ? q.substring(0, 25) + '...' : q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-[#6366F1]/60 transition-colors">
            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <p className="text-gray-600 mb-4 font-medium">点击或拖拽上传图像</p>
            <button className="btn-primary">选择图像</button>
            <p className="text-sm text-gray-500 mt-3">
              （图像上传功能开发中）
            </p>
          </div>
        )}

        {/* 搜索按钮 */}
        <button
          onClick={handleSearch}
          disabled={loading}
          className="btn-primary w-full mt-4 py-3 text-base"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              搜索中...
            </span>
          ) : (
            <span className="flex items-center justify-center">
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              开始检索
            </span>
          )}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 flex items-start">
          <svg className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {/* 搜索结果 */}
      {results.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <ResultsGrid results={results} searchTime={searchTime} />
        </motion.div>
      )}
    </div>
  );
}
