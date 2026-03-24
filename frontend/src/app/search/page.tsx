'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import ImageUpload from '@/components/ImageUpload';
import ResultsGrid from '@/components/ResultsGrid';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// 检索模式类型
type SearchMode = 'image_to_image' | 'label_to_image' | 'image_to_label';

// 搜索结果类型
interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  labels: number[];
  thumbnail_url?: string;
  hash_code?: number[];
}

// 图搜标签结果类型
interface ImageToLabelResult {
  rank: number;
  image_id: number;
  labels: number[];
  score: number;
  distance: number;
}

// 加密信息类型
interface EncryptionInfo {
  method: string;
  query_encrypted: boolean;
  database_encrypted: boolean;
  security_level: number;
  bit_dim: number;
}

// CIR 搜索结果类型
interface CIRSearchResult {
  rank: number;
  image_name: string;
  score: number;
}

// CIR 服务状态
interface CIRStatus {
  initialized: boolean;
  indexed: boolean;
  index_size: number;
  feature_dim: number;
  model_loaded: boolean;
  keys_loaded: boolean;
}

export default function UnifiedSearchPage() {
  // 检索模式
  const [mode, setMode] = useState<SearchMode>('label_to_image');
  const [useEncrypted, setUseEncrypted] = useState(true);
  const [topK, setTopK] = useState(10);

  // 查询输入
  const [selectedLabels, setSelectedLabels] = useState<number[]>([]);
  const [labelInput, setLabelInput] = useState('');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  // 结果状态
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [labelResults, setLabelResults] = useState<ImageToLabelResult[]>([]);
  const [cirResults, setCirResults] = useState<CIRSearchResult[]>([]);
  const [searchTime, setSearchTime] = useState(0);
  const [error, setError] = useState('');
  const [encryptionInfo, setEncryptionInfo] = useState<EncryptionInfo | null>(null);

  // 标签统计
  const [topLabels, setTopLabels] = useState<{ index: number; count: number }[]>([]);

  // CIR 服务状态
  const [cirStatus, setCirStatus] = useState<CIRStatus | null>(null);

  // 加载初始数据
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取标签统计
        const labelsRes = await axios.get(`${API_BASE}/api/labels/stats`);
        if (labelsRes.data.success) {
          setTopLabels(labelsRes.data.top_labels || []);
        }
      } catch (err) {
        console.error('获取标签信息失败:', err);
      }

      try {
        // 获取 CIR 状态
        const cirRes = await axios.get(`${API_BASE}/api/cir/status`);
        setCirStatus(cirRes.data);
      } catch (err) {
        console.error('获取 CIR 状态失败:', err);
      }
    };

    fetchData();
  }, []);

  // 解析标签输入
  const parseLabelInput = (input: string): number[] => {
    return input
      .split(',')
      .map((s) => parseInt(s.trim()))
      .filter((n) => !isNaN(n) && n >= 0);
  };

  // 切换标签选择
  const toggleLabel = (index: number) => {
    setSelectedLabels((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  // 执行检索
  const handleSearch = async () => {
    // 验证输入
    if (mode === 'label_to_image' && selectedLabels.length === 0) {
      setError('请选择至少一个标签');
      return;
    }

    if ((mode === 'image_to_label' || mode === 'image_to_image') && !uploadedImage) {
      setError('请上传查询图像');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);
    setLabelResults([]);
    setCirResults([]);
    setEncryptionInfo(null);

    try {
      if (mode === 'image_to_image') {
        // CIR 图搜图
        await handleCIRSearch();
      } else {
        // DCMH 跨模态检索
        await handleDCMHSearch();
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || '检索失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  // DCMH 跨模态检索
  const handleDCMHSearch = async () => {
    const response = await axios.post(`${API_BASE}/api/search`, {
      query_type: mode,
      label_indices: mode === 'label_to_image' ? selectedLabels : undefined,
      query_image: mode === 'image_to_label' ? uploadedImage : undefined,
      top_k: topK,
      use_encrypted: useEncrypted,
    });

    setSearchTime(response.data.search_time_ms || 0);
    setEncryptionInfo(response.data.encryption_info || null);

    if (mode === 'image_to_label') {
      setLabelResults(response.data.label_results || []);
    } else {
      setResults(response.data.results || []);
    }
  };

  // CIR 图搜图
  const handleCIRSearch = async () => {
    if (!uploadedImage) return;

    const response = await fetch(uploadedImage);
    const blob = await response.blob();
    const file = new File([blob], 'query.jpg', { type: 'image/jpeg' });

    const formData = new FormData();
    formData.append('image', file);

    const endpoint = useEncrypted
      ? `${API_BASE}/api/cir/sknn/search/upload`
      : `${API_BASE}/api/cir/search/upload`;

    const res = await axios.post(endpoint, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: { top_k: topK },
    });

    setSearchTime(0);
    setEncryptionInfo(
      useEncrypted
        ? {
            method: 'SkNN',
            query_encrypted: true,
            database_encrypted: true,
            security_level: 2,
            bit_dim: cirStatus?.feature_dim || 2048,
          }
        : null
    );
    setCirResults(res.data.results || []);
  };

  // 图像上传回调
  const handleImageSelected = (file: File) => {
    console.log('图像已选择:', file.name);
  };

  const handleImageUploaded = (imageUrl: string) => {
    setUploadedImage(imageUrl);
  };

  // 获取模式标题
  const getModeTitle = () => {
    switch (mode) {
      case 'image_to_image':
        return '图搜图 (CIR)';
      case 'label_to_image':
        return '标签搜图 (DCMH T2I)';
      case 'image_to_label':
        return '图搜标签 (DCMH I2T)';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">统一检索系统</h1>
        <p className="text-gray-600">
          整合 CIR 图搜图与 DCMH 跨模态检索，支持隐私保护检索模式
        </p>
      </div>

      {/* 检索模式卡片 */}
      <div className="card mb-8">
        {/* Tab 切换 */}
        <div className="flex space-x-2 mb-6 p-1 bg-gray-100 rounded-xl w-fit">
          <button
            onClick={() => setMode('label_to_image')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              mode === 'label_to_image'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                />
              </svg>
              <span>标签搜图</span>
            </div>
          </button>
          <button
            onClick={() => setMode('image_to_label')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              mode === 'image_to_label'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                />
              </svg>
              <span>图搜标签</span>
            </div>
          </button>
          <button
            onClick={() => setMode('image_to_image')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              mode === 'image_to_image'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span>图搜图</span>
            </div>
          </button>
        </div>

        {/* 查询输入区域 */}
        {mode === 'label_to_image' ? (
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
                <p className="text-sm text-gray-600 mb-2">
                  已选择 {selectedLabels.length} 个标签：
                </p>
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
                        <svg
                          className="w-3 h-3"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                          />
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

        {/* 检索选项 */}
        <div className="flex items-center gap-6 mt-4 mb-4">
          {/* 检索模式 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">检索模式：</span>
            <div className="flex space-x-2 p-1 bg-gray-100 rounded-lg">
              <button
                onClick={() => setUseEncrypted(true)}
                className={`px-3 py-1 rounded-md text-sm transition-all ${
                  useEncrypted
                    ? 'bg-white text-[#6366F1] shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                隐私保护
              </button>
              <button
                onClick={() => setUseEncrypted(false)}
                className={`px-3 py-1 rounded-md text-sm transition-all ${
                  !useEncrypted
                    ? 'bg-white text-[#6366F1] shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                明文
              </button>
            </div>
          </div>

          {/* Top-K */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Top-K：</span>
            <select
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="input-primary w-24"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>

        {/* 搜索按钮 */}
        <button
          onClick={handleSearch}
          disabled={loading}
          className="btn-primary w-full py-3 text-base"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              检索中...
            </span>
          ) : (
            <span className="flex items-center justify-center">
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              开始检索
            </span>
          )}
        </button>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 flex items-start">
          <svg
            className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
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
                <svg
                  className="w-5 h-5 text-[#6366F1]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
              </div>
              <div>
                <p className="font-medium text-gray-800">隐私保护检索已启用</p>
                <p className="text-sm text-gray-600">
                  {encryptionInfo.method} · {encryptionInfo.bit_dim} 位
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-1">
                <span
                  className={`w-2 h-2 rounded-full ${
                    encryptionInfo.query_encrypted ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                />
                <span className="text-gray-600">查询加密</span>
              </div>
              <div className="flex items-center space-x-1">
                <span
                  className={`w-2 h-2 rounded-full ${
                    encryptionInfo.database_encrypted ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                />
                <span className="text-gray-600">数据库加密</span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 检索结果 */}
      {results.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <ResultsGrid results={results} searchTime={searchTime} />
        </motion.div>
      )}

      {/* 图搜标签结果 */}
      {labelResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            检索结果 - 相关标签
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {labelResults.map((result) => (
              <div
                key={result.rank}
                className="bg-gray-50 rounded-lg p-4 text-center"
              >
                <div className="text-sm font-medium text-gray-800 mb-1">
                  来源图像: {result.image_id}
                </div>
                <div className="flex flex-wrap gap-1 justify-center mb-2">
                  {result.labels.slice(0, 5).map((label) => (
                    <span
                      key={label}
                      className="px-2 py-0.5 bg-[#6366F1]/10 text-[#6366F1] rounded text-xs"
                    >
                      {label}
                    </span>
                  ))}
                  {result.labels.length > 5 && (
                    <span className="text-xs text-gray-500">
                      +{result.labels.length - 5}
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-500">
                  得分: {result.score.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* CIR 图搜图结果 */}
      {cirResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            检索结果 - 相似图像
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {cirResults.map((result) => (
              <div
                key={result.rank}
                className="bg-gray-50 rounded-lg p-4 text-center"
              >
                <div className="text-sm font-medium text-gray-800 mb-1 truncate">
                  {result.image_name}
                </div>
                <div className="text-xs text-gray-500">
                  排名: {result.rank} | 得分: {result.score.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* 使用说明 */}
      <div className="mt-8 card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">检索模式说明</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <svg
                className="w-5 h-5 text-[#6366F1]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                />
              </svg>
              <span className="font-medium">标签搜图 (DCMH T2I)</span>
            </div>
            <p className="text-sm text-gray-600">
              选择标签索引，检索具有相似标签的图像。基于 DCMH 深度跨模态哈希模型。
            </p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <svg
                className="w-5 h-5 text-[#6366F1]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                />
              </svg>
              <span className="font-medium">图搜标签 (DCMH I2T)</span>
            </div>
            <p className="text-sm text-gray-600">
              上传图像，预测图像的相关标签。跨模态检索的图像→文本方向。
            </p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <svg
                className="w-5 h-5 text-[#6366F1]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span className="font-medium">图搜图 (CIR)</span>
            </div>
            <p className="text-sm text-gray-600">
              上传图像，检索视觉相似图像。基于 CNN 特征和 SkNN 隐私保护检索。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}