'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import ImageUpload from '@/components/ImageUpload';
import ResultsGrid from '@/components/ResultsGrid';
import DatasetSelector from '@/components/DatasetSelector';
import TagSelector from '@/components/TagSelector';
import {
  SearchMode,
  EncryptionMode,
  DCMHDataset,
  CIRDataset,
  unifiedSearch,
  unifiedSearchUpload,
  getUnifiedStatus,
  getTagNames,
  ServiceStatus,
  TagToImageResult,
  ImageToTagResult,
  ImageToImageResult,
  EncryptionInfo,
  HitStats,
} from '@/lib/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function UnifiedSearchPage() {
  // 检索模式
  const [mode, setMode] = useState<SearchMode>('tag_to_image');
  const [encryption, setEncryption] = useState<EncryptionMode>('encrypted');
  const [topK, setTopK] = useState(10);

  // 数据集选择
  const [dcmhDataset, setDcmhDataset] = useState<DCMHDataset>('flickr25k');
  const [cirDataset, setCirDataset] = useState<CIRDataset>('roxford5k');

  // 服务状态
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);

  // 查询输入
  const [selectedTags, setSelectedTags] = useState<number[]>([]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // 结果状态
  const [loading, setLoading] = useState(false);
  const [tagToImageResults, setTagToImageResults] = useState<TagToImageResult[]>([]);
  const [imageToTagResults, setImageToTagResults] = useState<ImageToTagResult[]>([]);
  const [imageToImageResults, setImageToImageResults] = useState<ImageToImageResult[]>([]);
  const [error, setError] = useState('');
  const [encryptionInfo, setEncryptionInfo] = useState<EncryptionInfo | null>(null);
  const [hitStats, setHitStats] = useState<HitStats | null>(null);
  const [searchTimeMs, setSearchTimeMs] = useState(0);

  // 标签名称
  const [tagNames, setTagNames] = useState<string[]>([]);

  // 加载标签名称
  useEffect(() => {
    const loadTagNames = async () => {
      try {
        const result = await getTagNames(dcmhDataset);
        if (result.success) {
          setTagNames(result.tag_names);
        }
      } catch (err) {
        console.error('加载标签名称失败:', err);
      }
    };

    if (mode !== 'image_to_image') {
      loadTagNames();
    }
  }, [dcmhDataset, mode]);

  // 获取服务状态
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const currentDataset = mode === 'image_to_image' ? cirDataset : dcmhDataset;
        const status = await getUnifiedStatus(currentDataset);

        if (mode === 'image_to_image') {
          setServiceStatus(status.cir_status || null);
        } else {
          setServiceStatus(status.dcmh_status || null);
        }
      } catch (err) {
        console.error('获取服务状态失败:', err);
      }
    };

    fetchStatus();
  }, [dcmhDataset, cirDataset, mode]);

  // 执行检索
  const handleSearch = async () => {
    // 验证输入
    if (mode === 'tag_to_image' && selectedTags.length === 0) {
      setError('请选择至少一个标签');
      return;
    }

    if ((mode === 'image_to_tag' || mode === 'image_to_image') && !uploadedFile) {
      setError('请上传查询图像');
      return;
    }

    setLoading(true);
    setError('');
    setTagToImageResults([]);
    setImageToTagResults([]);
    setImageToImageResults([]);
    setEncryptionInfo(null);
    setHitStats(null);

    try {
      const currentDataset = mode === 'image_to_image' ? cirDataset : dcmhDataset;

      let response;

      if (mode === 'tag_to_image') {
        // 标签搜图（JSON 请求）
        response = await unifiedSearch({
          mode: mode,
          encryption: encryption,
          dataset: currentDataset,
          top_k: topK,
          tag_indices: selectedTags,
        });
      } else if (uploadedFile) {
        // 图像检索（文件上传）
        response = await unifiedSearchUpload(
          uploadedFile,
          mode,
          encryption,
          currentDataset,
          topK
        );
      }

      if (response) {
        setSearchTimeMs(response.search_time_ms);
        setEncryptionInfo(response.encryption_info || null);
        setHitStats(response.hit_stats || null);

        if (mode === 'tag_to_image') {
          setTagToImageResults(response.results as TagToImageResult[]);
        } else if (mode === 'image_to_tag') {
          setImageToTagResults(response.tag_results);
        } else {
          setImageToImageResults(response.results as ImageToImageResult[]);
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || '检索失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  // 图像上传回调
  const handleImageSelected = (file: File) => {
    setUploadedFile(file);
  };

  const handleImageUploaded = (imageUrl: string) => {
    setUploadedImage(imageUrl);
  };

  const handleImageCleared = () => {
    setUploadedImage(null);
    setUploadedFile(null);
  };

  // 获取当前数据集
  const getCurrentDataset = () => mode === 'image_to_image' ? cirDataset : dcmhDataset;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">统一检索系统</h1>
        <p className="text-gray-600">
          支持标签搜图、图搜标签和图搜图三种检索模式，可选隐私保护检索
        </p>
      </div>

      {/* 检索模式卡片 */}
      <div className="card mb-8">
        {/* Tab 切换 */}
        <div className="flex flex-wrap items-center gap-4 mb-6">
          <div className="flex space-x-2 p-1 bg-gray-100 rounded-xl">
            <button
              onClick={() => setMode('tag_to_image')}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                mode === 'tag_to_image'
                  ? 'bg-white text-[#6366F1] shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                </svg>
                <span>标签搜图</span>
              </div>
            </button>
            <button
              onClick={() => setMode('image_to_tag')}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                mode === 'image_to_tag'
                  ? 'bg-white text-[#6366F1] shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <div className="flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>图搜图</span>
              </div>
            </button>
          </div>

          {/* 数据集选择器 */}
          {mode === 'image_to_image' ? (
            <DatasetSelector
              type="cir"
              value={cirDataset}
              onChange={(v) => setCirDataset(v as CIRDataset)}
            />
          ) : (
            <DatasetSelector
              type="dcmh"
              value={dcmhDataset}
              onChange={(v) => setDcmhDataset(v as DCMHDataset)}
            />
          )}
        </div>

        {/* 服务状态 */}
        {serviceStatus && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${serviceStatus.initialized ? 'bg-green-500' : 'bg-yellow-500'}`} />
              <span className="text-gray-600">
                {serviceStatus.initialized ? '服务就绪' : '服务初始化中'}
              </span>
            </div>
            <div className="text-gray-500">
              索引大小: {serviceStatus.index_size}
            </div>
            {serviceStatus.keys_loaded && (
              <div className="text-emerald-600">
                密钥已加载
              </div>
            )}
          </div>
        )}

        {/* 查询输入区域 */}
        {mode === 'tag_to_image' ? (
          <TagSelector
            dataset={dcmhDataset}
            selectedTags={selectedTags}
            onChange={setSelectedTags}
            placeholder="搜索并选择标签..."
          />
        ) : (
          <ImageUpload
            onImageSelected={handleImageSelected}
            onImageUploaded={handleImageUploaded}
            onImageCleared={handleImageCleared}
          />
        )}

        {/* 检索选项 */}
        <div className="flex items-center gap-6 mt-4 mb-4">
          {/* 加密模式 */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">检索模式：</span>
            <div className="flex space-x-2 p-1 bg-gray-100 rounded-lg">
              <button
                onClick={() => setEncryption('encrypted')}
                className={`px-3 py-1 rounded-md text-sm transition-all ${
                  encryption === 'encrypted'
                    ? 'bg-white text-[#6366F1] shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                隐私保护
              </button>
              <button
                onClick={() => setEncryption('plaintext')}
                className={`px-3 py-1 rounded-md text-sm transition-all ${
                  encryption === 'plaintext'
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
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              检索中...
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
      {encryptionInfo && (encryptionInfo.query_encrypted || encryptionInfo.database_encrypted) && (
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
                  {encryptionInfo.method} · {encryptionInfo.bit_dim || encryptionInfo.feature_dim} 维
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
        </motion.div>
      )}

      {/* 标签搜图结果 */}
      {tagToImageResults.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <ResultsGrid
            results={tagToImageResults.map(r => ({
              rank: r.rank,
              image_id: parseInt(r.image_id),
              score: r.score,
              distance: r.distance,
              tags: r.tags,
              tag_names: r.tag_names,
              hit_tags: r.hit_tags,
              hit_tag_names: r.hit_tag_names,
              thumbnail_url: r.thumbnail_url,
              hash_code: r.hash_code,
              category_hit: r.category_hit,
              tag_hit: r.tag_hit,
              category_names: r.category_names,
              hit_category_names: r.hit_category_names,
            }))}
            searchTime={searchTimeMs}
            hitStats={hitStats}
          />
        </motion.div>
      )}

      {/* 图搜标签结果 */}
      {imageToTagResults.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            检索结果 - 相关标签
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {imageToTagResults.map((result) => (
              <div key={result.rank} className="bg-gray-50 rounded-lg p-4 text-center">
                {result.thumbnail_url && (
                  <div className="mb-3">
                    <img
                      src={`${API_BASE}${result.thumbnail_url}`}
                      alt={`来源图像 ${result.image_id}`}
                      className="w-full h-24 object-cover rounded-lg mx-auto"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                )}
                <div className="text-sm font-medium text-gray-800 mb-1">
                  来源图像: {result.image_id}
                </div>
                <div className="flex flex-wrap gap-1 justify-center mb-2">
                  {result.tag_names.slice(0, 5).map((tag, idx) => (
                    <span key={idx} className="px-2 py-0.5 bg-[#6366F1]/10 text-[#6366F1] rounded text-xs">
                      {tag}
                    </span>
                  ))}
                  {result.tag_names.length > 5 && (
                    <span className="text-xs text-gray-500">+{result.tag_names.length - 5}</span>
                  )}
                </div>
                {result.category_names && result.category_names.length > 0 && (
                  <div className="pt-2 border-t border-gray-200 mt-2">
                    <p className="text-xs text-gray-500 mb-1">类别</p>
                    <div className="flex flex-wrap gap-1 justify-center">
                      {result.category_names.slice(0, 4).map((name, i) => (
                        <span key={i} className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="text-xs text-gray-500 mt-2">
                  得分: {result.score.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* 图搜图结果 */}
      {imageToImageResults.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <div className="mb-4 flex items-center justify-between">
            <span className="text-sm text-gray-600">
              检索完成，耗时 <span className="font-mono font-semibold text-[#6366F1]">{searchTimeMs.toFixed(2)}</span> ms
            </span>
            <span className="badge-primary">共 {imageToImageResults.length} 个结果</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {imageToImageResults.map((result) => (
              <div key={result.rank} className="card group cursor-pointer hover:shadow-lg transition-all">
                <div className="relative bg-gray-100 rounded-xl h-40 mb-3 overflow-hidden">
                  {result.thumbnail_url ? (
                    <img
                      src={`${API_BASE}${result.thumbnail_url}`}
                      alt={`结果 ${result.rank}`}
                      className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <span className="text-gray-400">图像 {result.image_name}</span>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">#{result.rank}</span>
                  <span className="text-sm text-[#6366F1] font-mono">{(result.score * 100).toFixed(1)}%</span>
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
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span className="font-medium">标签搜图</span>
            </div>
            <p className="text-sm text-gray-600">
              选择标签，检索具有相似标签的图像。基于 DCMH 深度跨模态哈希。
            </p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              <span className="font-medium">图搜标签</span>
            </div>
            <p className="text-sm text-gray-600">
              上传图像，检索该图像的相关标签。基于 DCMH 深度跨模态哈希。
            </p>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span className="font-medium">图搜图</span>
            </div>
            <p className="text-sm text-gray-600">
              上传图像，检索相似图像。基于 CNN 特征的图像检索。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}