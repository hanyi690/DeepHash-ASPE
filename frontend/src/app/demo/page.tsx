'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import ResultsGrid from '@/components/ResultsGrid';
import ImageUpload from '@/components/ImageUpload';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  labels: number[];
  thumbnail_url?: string;
  hash_code?: number[];
}

interface EncryptionInfo {
  method: string;
  query_encrypted: boolean;
  database_encrypted: boolean;
  security_level: number;
  bit_dim: number;
}

interface SearchResponse {
  success: boolean;
  query_type: string;
  label_indices?: number[];
  results: SearchResult[];
  total_results: number;
  search_time_ms: number;
  encryption_info?: EncryptionInfo;
  query_hash_code?: number[];
}

export default function DemoPage() {
  const [queryType, setQueryType] = useState<'label' | 'image'>('label');
  const [selectedLabels, setSelectedLabels] = useState<number[]>([]);
  const [labelInput, setLabelInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchTime, setSearchTime] = useState(0);
  const [error, setError] = useState('');
  const [encryptionInfo, setEncryptionInfo] = useState<EncryptionInfo | null>(null);
  const [queryHashCode, setQueryHashCode] = useState<number[] | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [topLabels, setTopLabels] = useState<{index: number; count: number}[]>([]);

  // 获取标签信息
  useEffect(() => {
    const fetchLabelStats = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/labels/stats`);
        if (response.data.success) {
          setTopLabels(response.data.top_labels || []);
        }
      } catch (err) {
        console.error('获取标签信息失败:', err);
      }
    };
    fetchLabelStats();
  }, []);

  // 解析标签输入
  const parseLabelInput = (input: string): number[] => {
    return input.split(',')
      .map(s => parseInt(s.trim()))
      .filter(n => !isNaN(n) && n >= 0);
  };

  // 切换标签选择
  const toggleLabel = (index: number) => {
    setSelectedLabels(prev =>
      prev.includes(index)
        ? prev.filter(i => i !== index)
        : [...prev, index]
    );
  };

  const handleSearch = async () => {
    if (queryType === 'label' && selectedLabels.length === 0) {
      setError('请选择至少一个标签');
      return;
    }

    if (queryType === 'image' && !uploadedImage) {
      setError('请上传查询图像');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post<SearchResponse>(`${API_BASE}/api/search`, {
        query_type: queryType,
        label_indices: queryType === 'label' ? selectedLabels : undefined,
        query_image: queryType === 'image' ? uploadedImage : undefined,
        top_k: 10,
        use_encrypted: true
      });

      setResults(response.data.results || []);
      setSearchTime(response.data.search_time_ms || 0);
      setEncryptionInfo(response.data.encryption_info || null);
      setQueryHashCode(response.data.query_hash_code || null);
    } catch (err: any) {
      setError(err.response?.data?.detail || '搜索失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const handleImageSelected = (file: File) => {
    console.log('图像已选择:', file.name);
  };

  const handleImageUploaded = (imageUrl: string) => {
    setUploadedImage(imageUrl);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">跨模态检索演示</h1>
        <p className="text-gray-600">
          选择标签索引或上传图像，系统将返回最相关的图像结果
        </p>
      </div>

      {/* 查询卡片 */}
      <div className="card mb-8">
        {/* 查询类型切换 */}
        <div className="flex space-x-2 mb-6 p-1 bg-gray-100 rounded-xl w-fit">
          <button
            onClick={() => setQueryType('label')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              queryType === 'label'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span>标签查询</span>
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
        {queryType === 'label' ? (
          <div>
            {/* 常用标签选择 */}
            <div className="mb-4">
              <p className="text-sm text-gray-500 mb-2">点击选择常用标签：</p>
              <div className="flex flex-wrap gap-2">
                {topLabels.slice(0, 20).map((label) => (
                  <button
                    key={label.index}
                    onClick={() => toggleLabel(label.index)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                      selectedLabels.includes(label.index)
                        ? 'bg-[#6366F1] text-white shadow-sm'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {label.index}
                  </button>
                ))}
              </div>
            </div>

            {/* 自定义标签输入 */}
            <div className="mb-4">
              <p className="text-sm text-gray-500 mb-2">或输入标签索引（逗号分隔）：</p>
              <input
                type="text"
                value={labelInput}
                onChange={(e) => {
                  setLabelInput(e.target.value);
                  setSelectedLabels(parseLabelInput(e.target.value));
                }}
                placeholder="例如：0, 5, 10, 20"
                className="input-primary"
              />
            </div>

            {/* 已选标签显示 */}
            {selectedLabels.length > 0 && (
              <div className="mb-4 p-3 bg-[#6366F1]/10 rounded-xl">
                <p className="text-sm text-gray-600 mb-2">已选择 {selectedLabels.length} 个标签：</p>
                <div className="flex flex-wrap gap-1">
                  {selectedLabels.map((idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center px-2 py-1 bg-white rounded-md text-sm"
                    >
                      {idx}
                      <button
                        onClick={() => toggleLabel(idx)}
                        className="ml-1 text-gray-400 hover:text-red-500"
                      >
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <ImageUpload
            onImageSelected={handleImageSelected}
            onImageUploaded={handleImageUploaded}
          />
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

      {/* 加密信息 */}
      {encryptionInfo && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-gradient-to-r from-[#6366F1]/10 to-[#8B5CF6]/10 border border-[#6366F1]/20 rounded-xl"
        >
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-[#6366F1]/20 flex items-center justify-center">
                <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <div>
                <p className="font-medium text-gray-800">隐私保护检索已启用</p>
                <p className="text-sm text-gray-600">
                  {encryptionInfo.method} · {encryptionInfo.bit_dim} 位哈希码
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-1">
                <span className={`w-2 h-2 rounded-full ${encryptionInfo.query_encrypted ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span className="text-gray-600">查询加密</span>
              </div>
              <div className="flex items-center space-x-1">
                <span className={`w-2 h-2 rounded-full ${encryptionInfo.database_encrypted ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span className="text-gray-600">数据库加密</span>
              </div>
            </div>
          </div>

          {/* 查询哈希码展示 */}
          {queryHashCode && (
            <div className="mt-4 pt-4 border-t border-[#6366F1]/20">
              <details className="group">
                <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800 flex items-center">
                  <svg className="w-4 h-4 mr-1 transform group-open:rotate-90 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  查看查询哈希码
                </summary>
                <div className="mt-2 p-2 bg-white/50 rounded-lg overflow-x-auto">
                  <code className="text-xs text-gray-700 font-mono">
                    [{queryHashCode.slice(0, 16).join(', ')}...]
                  </code>
                </div>
              </details>
            </div>
          )}
        </motion.div>
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