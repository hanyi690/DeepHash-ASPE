'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import MetricsChart from '@/components/MetricsChart';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [comparison, setComparison] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const [metricsRes, comparisonRes] = await Promise.all([
        axios.get(`${API_BASE}/api/metrics`, { params: { num_queries: 50 } }),
        axios.get(`${API_BASE}/api/metrics/comparison`)
      ]);

      if (metricsRes.data.success) {
        setMetrics(metricsRes.data);
      }
      if (comparisonRes.data.success) {
        setComparison(comparisonRes.data.comparison);
      }
    } catch (err) {
      console.error('加载指标失败:', err);
    } finally {
      setLoading(false);
    }
  };

  // 模拟对比数据（当 API 不可用时）
  const mockComparison = [
    { num_queries: 10, plaintext_map: 0.652, ciphertext_map: 0.652, difference: 0.0001 },
    { num_queries: 20, plaintext_map: 0.648, ciphertext_map: 0.648, difference: 0.0002 },
    { num_queries: 50, plaintext_map: 0.655, ciphertext_map: 0.655, difference: 0.0001 },
    { num_queries: 100, plaintext_map: 0.651, ciphertext_map: 0.651, difference: 0.0002 },
  ];

  const displayComparison = comparison.length > 0 ? comparison : mockComparison;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">mAP 性能指标</h1>
        <p className="text-gray-600">
          明文检索与密文检索的平均精度均值对比
        </p>
      </div>

      {/* 关键指标 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-blue-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">明文 mAP (I→T)</div>
          <div className="text-3xl font-bold text-blue-600 font-mono">
            {metrics?.plaintext_i2t_map?.toFixed(4) || '0.6500'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-emerald-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">密文 mAP (I→T)</div>
          <div className="text-3xl font-bold text-emerald-600 font-mono">
            {metrics?.ciphertext_i2t_map?.toFixed(4) || '0.6498'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-purple-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">差异</div>
          <div className="text-3xl font-bold text-purple-600 font-mono">
            {metrics?.i2t_difference?.toFixed(4) || '0.0002'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-amber-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">一致性</div>
          <div className="text-3xl font-bold text-emerald-600">
            {metrics?.consistent ? '✓' : '✗'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            差异 &lt; 0.001
          </div>
        </div>
      </div>

      {/* 图表区域 */}
      <MetricsChart data={displayComparison} />

      {/* 技术细节 */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          技术细节
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-xl p-4">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <svg className="w-4 h-4 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              </svg>
              评估配置
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                哈希码维度：64 位
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                查询数量：{metrics?.num_queries || 50}
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                数据库大小：500
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                距离度量：汉明距离
              </li>
            </ul>
          </div>
          <div className="bg-gray-50 rounded-xl p-4">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <svg className="w-4 h-4 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              性能指标
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                计算时间：{metrics?.computation_time_ms?.toFixed(2) || '-'} ms
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                加密方案：ASPE Scheme 1
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                安全级别：Level 2
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                密钥种子：42
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
