/**
 * API 客户端
 *
 * 封装所有与后端的 API 调用
 */

import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// 创建 axios 实例
const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============== 类型定义 ==============

export type DCMHDataset = 'flickr25k' | 'nuswide';
export type CIRDataset = 'roxford5k' | 'rparis6k';

export interface DatasetStatus {
  name: string;
  cache_loaded: boolean;
  cache_path: string;
  model_loaded: boolean;
  gpu_available: boolean;
  gpu_name: string | null;
  database_size: number;
  query_size: number;
  bit_dim: number;
}

export interface DatasetStatusResponse {
  success: boolean;
  status: DatasetStatus;
  message: string;
}

export interface SystemStatusResponse {
  success: boolean;
  gpu_available: boolean;
  gpu_name: string | null;
  device: string;
  datasets: Record<string, DatasetStatus>;
}

// DCMH 检索相关类型
export interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  labels: number[];  // YAll 索引
  label_names: string[];  // 标签名称
  hit_labels: number[];  // 命中的查询标签索引
  hit_label_names: string[];  // 命中的查询标签名称
  thumbnail_url?: string;
  hash_code?: number[];
}

export interface ImageToLabelResult {
  rank: number;
  image_id: number;
  labels: number[];  // YAll 索引
  label_names: string[];  // 标签名称
  score: number;
  distance: number;
}

export interface HitStats {
  total_results: number;
  hits: number;
  hit_rate: number;
  query_label_count: number;
  query_labels: number[];
  query_label_names: string[];
}

export interface EncryptionInfo {
  method: string;
  query_encrypted: boolean;
  database_encrypted: boolean;
  security_level: number;
  bit_dim: number;
}

export interface UnifiedSearchResponse {
  success: boolean;
  query_type: string;
  label_indices?: number[];
  query_label_names?: string[];  // 查询标签名称列表
  results: SearchResult[];
  label_results: ImageToLabelResult[];
  total_results: number;
  search_time_ms: number;
  hit_stats?: HitStats;  // 命中率统计
  encryption_info?: EncryptionInfo;
  query_hash_code?: number[];
}

// CIR 检索相关类型
export interface CIRSearchResult {
  rank: number;
  image_name: string;
  score: number;
  image_url?: string;
}

export interface CIRStatus {
  initialized: boolean;
  indexed: boolean;
  index_size: number;
  feature_dim: number;
  model_loaded: boolean;
  keys_loaded: boolean;
}

export interface SknnStatus {
  keys_generated: boolean;
  database_loaded: boolean;
  database_size: number;
}

// 标签统计类型
export interface LabelStats {
  success: boolean;
  total: number;
  label_dim: number;
  top_labels: { index: number; count: number }[];
}

// ============== API 函数 ==============

// 数据集状态
export async function getDatasetStatus(name: string): Promise<DatasetStatusResponse> {
  const response = await api.get(`/api/datasets/${name}/status`);
  return response.data;
}

// 系统状态
export async function getSystemStatus(): Promise<SystemStatusResponse> {
  const response = await api.get('/api/datasets/system/status');
  return response.data;
}
