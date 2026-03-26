'use client';

import { useState } from 'react';

interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  labels: number[];
  label_names: string[];  // 标签名称
  hit_labels: number[];  // 命中的查询标签索引
  hit_label_names: string[];  // 命中的查询标签名称
  thumbnail_url?: string;
  hash_code?: number[];
}

interface HitStats {
  total_results: number;
  hits: number;
  hit_rate: number;
  query_label_count: number;
  query_labels: number[];
  query_label_names: string[];
}

interface ResultsGridProps {
  results: SearchResult[];
  searchTime?: number;
  hitStats?: HitStats;  // 命中率统计
}

export default function ResultsGrid({ results, searchTime, hitStats }: ResultsGridProps) {
  const [expandedHash, setExpandedHash] = useState<number | null>(null);

  if (results.length === 0) {
    return null;
  }

  const toggleHash = (index: number) => {
    setExpandedHash(expandedHash === index ? null : index);
  };

  return (
    <div className="w-full">
      {/* 搜索结果统计 */}
      {searchTime !== undefined && (
        <div className="mb-6 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-sm text-gray-600">
                检索完成，耗时 <span className="font-mono font-semibold text-[#6366F1]">{searchTime.toFixed(2)}</span> ms
              </span>
            </div>
            {hitStats && (
              <div className="flex items-center space-x-2 text-sm">
                <span className="text-gray-600">命中率:</span>
                <span className={`font-bold ${
                  hitStats.hit_rate >= 0.8 ? 'text-emerald-600' :
                  hitStats.hit_rate >= 0.5 ? 'text-amber-600' : 'text-red-600'
                }`}>
                  {(hitStats.hit_rate * 100).toFixed(1)}%
                </span>
                <span className="text-gray-500">
                  ({hitStats.hits}/{hitStats.total_results})
                </span>
              </div>
            )}
          </div>
          <span className="badge-primary">
            共 {results.length} 个结果
          </span>
        </div>
      )}

      {/* 查询标签显示 */}
      {hitStats && hitStats.query_label_names && hitStats.query_label_names.length > 0 && (
        <div className="mb-4 p-3 bg-[#6366F1]/5 border border-[#6366F1]/20 rounded-lg">
          <div className="flex items-center flex-wrap gap-2">
            <span className="text-sm text-gray-600">查询标签:</span>
            {hitStats.query_label_names.map((name, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-[#6366F1]/10 text-[#6366F1] rounded text-sm font-medium"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 结果网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((result, index) => (
          <div
            key={index}
            className="card group cursor-pointer hover:shadow-lg hover:-translate-y-1 transition-all duration-200"
          >
            {/* 头部：排名和图像 ID */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                  ${result.rank === 1 ? 'bg-[#6366F1] text-white' : 'bg-gray-100 text-gray-600'}`}>
                  {result.rank}
                </span>
                <span className="text-sm font-medium text-gray-500">
                  图像 {result.image_id}
                </span>
              </div>
              <span className="badge-neutral font-mono text-xs">
                距离：{result.distance.toFixed(2)}
              </span>
            </div>

            {/* 图像区域 */}
            <div className="relative bg-gray-100 rounded-xl h-40 mb-3 overflow-hidden">
              {result.thumbnail_url ? (
                <img
                  src={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${result.thumbnail_url}${result.thumbnail_url.startsWith('/api/images/') ? '?format=image' : ''}`}
                  alt={`结果 ${result.rank}`}
                  className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300"
                  onError={(e) => {
                    // 图像加载失败时显示占位符
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <svg className="w-12 h-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-gray-400 ml-2">图像 {result.image_id}</span>
                </div>
              )}
              {/* 悬停覆盖层 */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
            </div>

            {/* 相似度进度条 */}
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-gray-500">相似度</span>
                <span className={`text-sm font-bold font-mono
                  ${result.score >= 0.8 ? 'text-emerald-600' : result.score >= 0.6 ? 'text-[#6366F1]' : 'text-amber-600'}`}>
                  {Math.min(Math.max(result.score * 100, 0), 100).toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className={`progress-bar-fill ${
                    result.score >= 0.8 ? 'from-emerald-500 to-emerald-400' :
                    result.score >= 0.6 ? 'from-[#6366F1] to-[#818CF8]' :
                    'from-amber-500 to-amber-400'
                  }`}
                  style={{ width: `${Math.min(Math.max(result.score * 100, 0), 100)}%` }}
                />
              </div>
            </div>

            {/* 哈希码展示 */}
            {result.hash_code && (
              <div className="mb-3">
                <button
                  onClick={(e) => { e.stopPropagation(); toggleHash(index); }}
                  className="w-full flex items-center justify-between p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center space-x-2 text-xs text-gray-600">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    <span>哈希码 ({result.hash_code.length} 位)</span>
                  </div>
                  <svg
                    className={`w-4 h-4 text-gray-400 transform transition-transform ${expandedHash === index ? 'rotate-180' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {expandedHash === index && (
                  <div className="mt-2 p-2 bg-gray-900 rounded-lg overflow-x-auto">
                    <code className="text-xs text-emerald-400 font-mono whitespace-nowrap">
                      [{result.hash_code.join(', ')}]
                    </code>
                  </div>
                )}
              </div>
            )}

            {/* 标签名称显示 */}
            {result.label_names && result.label_names.length > 0 && (
              <div className="pt-3 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-2 flex items-center">
                  <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                  </svg>
                  标签
                </p>
                <div className="flex flex-wrap gap-1">
                  {result.label_names.slice(0, 10).map((name, i) => {
                    // 检查是否为命中标签
                    const isHit = result.hit_label_names && result.hit_label_names.includes(name);
                    return (
                      <span
                        key={i}
                        className={`px-2 py-0.5 rounded text-xs font-medium ${
                          isHit
                            ? 'bg-emerald-100 text-emerald-700 ring-1 ring-emerald-300'
                            : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {name}
                        {isHit && ' ✓'}
                      </span>
                    );
                  })}
                  {result.label_names.length > 10 && (
                    <span className="px-2 py-0.5 text-gray-400 text-xs">
                      +{result.label_names.length - 10} 更多
                    </span>
                  )}
                </div>
                {/* 命中统计 */}
                {result.hit_label_names && result.hit_label_names.length > 0 && (
                  <p className="text-xs text-emerald-600 mt-2">
                    命中 {result.hit_label_names.length} 个查询标签
                  </p>
                )}
              </div>
            )}

            {/* 标签索引（折叠显示） */}
            {result.labels && result.labels.length > 0 && (
              <div className="pt-3 border-t border-gray-100">
                <details className="text-xs">
                  <summary className="text-gray-500 cursor-pointer hover:text-gray-700">
                    标签索引 (YAll): {result.labels.length} 个
                  </summary>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {result.labels.slice(0, 20).map((label, i) => (
                      <span
                        key={i}
                        className={`px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-mono ${
                          result.hit_labels && result.hit_labels.includes(label)
                            ? 'ring-1 ring-emerald-400'
                            : ''
                        }`}
                      >
                        {label}
                      </span>
                    ))}
                    {result.labels.length > 20 && (
                      <span className="px-2 py-0.5 text-gray-400 text-xs">
                        +{result.labels.length - 20}
                      </span>
                    )}
                  </div>
                </details>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
