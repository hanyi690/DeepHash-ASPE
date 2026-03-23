'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import ImageUpload from '@/components/ImageUpload';
import ResultsGrid from '@/components/ResultsGrid';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CIRSearchResult {
  rank: number;
  image_name: string;
  score: number;
}

interface CIRStatus {
  initialized: boolean;
  indexed: boolean;
  index_size: number;
  feature_dim: number;
  model_loaded: boolean;
  keys_loaded: boolean;
}

interface SknnStatus {
  keys_generated: boolean;
  database_loaded: boolean;
  database_size: number;
}

export default function CIRPage() {
  // 检索模式
  const [mode, setMode] = useState<'plaintext' | 'encrypted'>('encrypted');

  // 服务状态
  const [cirStatus, setCirStatus] = useState<CIRStatus | null>(null);
  const [sknnStatus, setSknnStatus] = useState<SknnStatus | null>(null);

  // 检索相关
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<CIRSearchResult[]>([]);
  const [error, setError] = useState('');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  // 数据库相关
  const [dbPath, setDbPath] = useState('');
  const [imageDir, setImageDir] = useState('');
  const [buildingDb, setBuildingDb] = useState(false);

  // 加载服务状态
  const loadStatus = async () => {
    try {
      const [cirRes, sknnRes] = await Promise.all([
        axios.get(`${API_BASE}/api/cir/status`),
        axios.get(`${API_BASE}/api/cir/sknn/status`)
      ]);
      setCirStatus(cirRes.data);
      setSknnStatus(sknnRes.data);
    } catch (err) {
      console.error('Failed to load status:', err);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  // 初始化服务
  const handleInitialize = async () => {
    try {
      await axios.get(`${API_BASE}/api/cir/initialize`, {
        params: {
          architecture: 'resnet101',
          pooling: 'gem',
          whitening: true
        }
      });
      loadStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || '初始化失败');
    }
  };

  // 生成 SkNN 密钥
  const handleGenerateKeys = async () => {
    try {
      await axios.post(`${API_BASE}/api/cir/sknn/keys/generate`, null, {
        params: { feature_dim: 2048 }
      });
      loadStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || '密钥生成失败');
    }
  };

  // 构建加密数据库
  const handleBuildDatabase = async () => {
    if (!imageDir || !dbPath) {
      setError('请输入图像目录和保存路径');
      return;
    }

    setBuildingDb(true);
    setError('');

    try {
      await axios.post(`${API_BASE}/api/cir/sknn/database/build`, null, {
        params: {
          image_dir: imageDir,
          save_dir: dbPath,
          feature_dim: 2048
        }
      });
      loadStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || '数据库构建失败');
    } finally {
      setBuildingDb(false);
    }
  };

  // 加载数据库
  const handleLoadDatabase = async () => {
    if (!dbPath) {
      setError('请输入数据库路径');
      return;
    }

    try {
      await axios.post(`${API_BASE}/api/cir/sknn/database/load`, null, {
        params: { db_dir: dbPath }
      });
      loadStatus();
    } catch (err: any) {
      setError(err.response?.data?.detail || '数据库加载失败');
    }
  };

  // 执行检索
  const handleSearch = async () => {
    if (!uploadedImage) {
      setError('请上传查询图像');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // 将 base64 转换为 File
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      const file = new File([blob], 'query.jpg', { type: 'image/jpeg' });

      const formData = new FormData();
      formData.append('image', file);

      const endpoint = mode === 'encrypted'
        ? `${API_BASE}/api/cir/sknn/search/upload`
        : `${API_BASE}/api/cir/search/upload`;

      const res = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { top_k: 10 }
      });

      setResults(res.data.results || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || '检索失败');
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
        <h1 className="text-3xl font-bold text-gray-900 mb-2">CNN 图像检索</h1>
        <p className="text-gray-600">
          基于 CNN 特征的图像检索系统，支持明文检索和 SkNN 隐私保护检索
        </p>
      </div>

      {/* 服务状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {/* CIR 服务状态 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">CIR 服务状态</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">初始化状态</span>
              <span className={`badge ${cirStatus?.initialized ? 'badge-success' : 'badge-neutral'}`}>
                {cirStatus?.initialized ? '已初始化' : '未初始化'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">模型加载</span>
              <span className={`badge ${cirStatus?.model_loaded ? 'badge-success' : 'badge-neutral'}`}>
                {cirStatus?.model_loaded ? '已加载' : '未加载'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">特征维度</span>
              <span className="text-gray-800">{cirStatus?.feature_dim || '-'}</span>
            </div>
          </div>
          <button
            onClick={handleInitialize}
            className="btn-secondary w-full mt-4"
          >
            初始化服务
          </button>
        </div>

        {/* SkNN 状态 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">SkNN 隐私检索状态</h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">密钥状态</span>
              <span className={`badge ${sknnStatus?.keys_generated ? 'badge-success' : 'badge-neutral'}`}>
                {sknnStatus?.keys_generated ? '已生成' : '未生成'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">数据库状态</span>
              <span className={`badge ${sknnStatus?.database_loaded ? 'badge-success' : 'badge-neutral'}`}>
                {sknnStatus?.database_loaded ? '已加载' : '未加载'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">数据库大小</span>
              <span className="text-gray-800">{sknnStatus?.database_size || 0} 张图像</span>
            </div>
          </div>
          <button
            onClick={handleGenerateKeys}
            className="btn-secondary w-full mt-4"
          >
            生成密钥
          </button>
        </div>
      </div>

      {/* 数据库管理 */}
      <div className="card mb-8">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">数据库管理</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">图像目录</label>
            <input
              type="text"
              value={imageDir}
              onChange={(e) => setImageDir(e.target.value)}
              placeholder="例如: data/flickr-25k/images"
              className="input-primary"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">数据库路径</label>
            <input
              type="text"
              value={dbPath}
              onChange={(e) => setDbPath(e.target.value)}
              placeholder="例如: data/cir_demo_db"
              className="input-primary"
            />
          </div>
        </div>
        <div className="flex space-x-4 mt-4">
          <button
            onClick={handleBuildDatabase}
            disabled={buildingDb}
            className="btn-primary flex-1"
          >
            {buildingDb ? '构建中...' : '构建数据库'}
          </button>
          <button
            onClick={handleLoadDatabase}
            className="btn-secondary flex-1"
          >
            加载数据库
          </button>
        </div>
      </div>

      {/* 检索区域 */}
      <div className="card mb-8">
        {/* 模式切换 */}
        <div className="flex space-x-2 mb-6 p-1 bg-gray-100 rounded-xl w-fit">
          <button
            onClick={() => setMode('encrypted')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              mode === 'encrypted'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
              <span>隐私保护检索</span>
            </div>
          </button>
          <button
            onClick={() => setMode('plaintext')}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              mode === 'plaintext'
                ? 'bg-white text-[#6366F1] shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span>明文检索</span>
            </div>
          </button>
        </div>

        {/* 图像上传 */}
        <ImageUpload
          onImageSelected={handleImageSelected}
          onImageUploaded={handleImageUploaded}
        />

        {/* 检索按钮 */}
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

      {/* 检索模式说明 */}
      {mode === 'encrypted' && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-gradient-to-r from-[#6366F1]/10 to-[#8B5CF6]/10 border border-[#6366F1]/20 rounded-xl"
        >
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full bg-[#6366F1]/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <p className="font-medium text-gray-800">SkNN 隐私保护检索</p>
              <p className="text-sm text-gray-600">
                基于 ASPE 加密方案，服务器无法获取明文特征，实现隐私保护的图像检索
              </p>
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
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">检索结果</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {results.map((result) => (
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
          </div>
        </motion.div>
      )}
    </div>
  );
}