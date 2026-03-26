/**
 * API 客户端
 *
 * 封装所有与后端的 API 调用
 * 统一 DCMH 和 CIR 检索接口
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

// ============== 统一类型定义 ==============

/** 检索模式 */
export type SearchMode = 'tag_to_image' | 'image_to_tag' | 'image_to_image';

/** 加密模式 */
export type EncryptionMode = 'plaintext' | 'encrypted';

/** DCMH 数据集 */
export type DCMHDataset = 'flickr25k' | 'nuswide';

/** CIR 数据集 */
export type CIRDataset = 'roxford5k' | 'rparis6k';

/** 所有数据集 */
export type Dataset = DCMHDataset | CIRDataset;

// ============== 统一结果类型 ==============

/** 基础检索结果 */
export interface BaseSearchResult {
  rank: number;
  image_id: string;
  score: number;
  distance: number;
  thumbnail_url?: string;
}

/** 标签搜图结果 */
export interface TagToImageResult extends BaseSearchResult {
  tags: number[];
  tag_names: string[];
  hit_tags: number[];
  hit_tag_names: string[];
  category_hit: boolean;
  tag_hit: boolean;
  category_names: string[];
  hit_category_names: string[];
  hash_code?: number[];
}

/** 图搜标签结果 */
export interface ImageToTagResult {
  rank: number;
  image_id: string;
  tags: number[];
  tag_names: string[];
  score: number;
  distance: number;
  thumbnail_url?: string;
  category_names?: string[];
}

/** 图搜图结果 */
export interface ImageToImageResult extends BaseSearchResult {
  image_name: string;
  image_url?: string;
}

/** 命中率统计 */
export interface HitStats {
  total_results: number;
  tag_hits: number;
  tag_hit_rate: number;
  category_hits: number;
  category_hit_rate: number;
  query_tags?: number[];
  query_tag_names?: string[];
  category_distribution?: Record<string, number>;
}

/** 加密信息 */
export interface EncryptionInfo {
  method: string;
  query_encrypted: boolean;
  database_encrypted: boolean;
  security_level: number;
  bit_dim?: number;
  feature_dim?: number;
}

/** 统一检索请求 */
export interface UnifiedSearchRequest {
  mode: SearchMode;
  encryption: EncryptionMode;
  dataset: string;
  top_k: number;
  tag_indices?: number[];
  query_image?: string;
}

/** 统一检索响应 */
export interface UnifiedSearchResponse {
  success: boolean;
  mode: SearchMode;
  encryption: EncryptionMode;
  dataset: string;
  results: BaseSearchResult[];
  tag_results: ImageToTagResult[];
  total_results: number;
  search_time_ms: number;
  hit_stats?: HitStats;
  query_tag_names?: string[];
  query_hash_code?: number[];
  encryption_info?: EncryptionInfo;
  message?: string;
}

/** 服务状态 */
export interface ServiceStatus {
  success: boolean;
  service_type: string;
  dataset: string;
  initialized: boolean;
  model_loaded: boolean;
  plaintext_indexed: boolean;
  encrypted_indexed: boolean;
  index_size: number;
  keys_loaded: boolean;
  additional_info?: Record<string, any>;
}

/** 统一状态响应 */
export interface UnifiedStatusResponse {
  success: boolean;
  dcmh_status?: ServiceStatus;
  cir_status?: ServiceStatus;
  key_manager_status?: Record<string, any>;
}

/** 标签统计 */
export interface TagStats {
  success: boolean;
  total: number;
  tag_dim: number;
  top_tags: { index: number; count: number }[];
}

// ============== 统一 API 函数 ==============

/**
 * 统一检索（JSON 请求）
 */
export async function unifiedSearch(request: UnifiedSearchRequest): Promise<UnifiedSearchResponse> {
  const response = await api.post('/api/search', request);
  return response.data;
}

/**
 * 统一检索（文件上传）
 */
export async function unifiedSearchUpload(
  file: File,
  mode: SearchMode,
  encryption: EncryptionMode,
  dataset: string,
  topK: number = 10
): Promise<UnifiedSearchResponse> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('mode', mode);
  formData.append('encryption', encryption);
  formData.append('dataset', dataset);
  formData.append('top_k', String(topK));

  const response = await api.post('/api/search/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

/**
 * 获取统一状态
 */
export async function getUnifiedStatus(dataset: string = 'flickr25k'): Promise<UnifiedStatusResponse> {
  const response = await api.get(`/api/search/status?dataset=${dataset}`);
  return response.data;
}

/**
 * 获取支持的数据集
 */
export async function getSupportedDatasets(): Promise<{
  dcmh_datasets: string[];
  cir_datasets: string[];
  all_datasets: string[];
  modes: Record<string, { description: string; supported_datasets: string[] }>;
}> {
  const response = await api.get('/api/search/datasets');
  return response.data;
}

/**
 * 重建缓存
 */
export async function rebuildCache(dataset: string = 'flickr25k'): Promise<{ success: boolean; message: string }> {
  const response = await api.post(`/api/search/rebuild-cache?dataset=${dataset}`);
  return response.data;
}

/**
 * 获取标签统计
 */
export async function getTagStats(dataset: DCMHDataset = 'flickr25k'): Promise<TagStats> {
  const response = await api.get(`/api/tags/stats?dataset=${dataset}`);
  return response.data;
}

/**
 * 获取标签名称
 */
export async function getTagNames(dataset: string): Promise<{ success: boolean; tag_names: string[] }> {
  const response = await api.get(`/api/tags/names/${dataset}`);
  return response.data;
}

/**
 * 获取图像
 */
export async function getImage(imageId: number, dataset: string = 'flickr25k'): Promise<{
  success: boolean;
  image_id: number;
  tags: number[];
  total_tags: number;
  thumbnail_url: string;
}> {
  const response = await api.get(`/api/images/${imageId}?dataset=${dataset}`);
  return response.data;
}

/**
 * 获取系统状态
 */
export async function getSystemStatus(): Promise<{
  success: boolean;
  gpu_available: boolean;
  gpu_name: string | null;
  device: string;
  datasets: Record<string, any>;
}> {
  const response = await api.get('/api/datasets/system/status');
  return response.data;
}

export default api;