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
  tags: number[];  // YAll 索引
  tag_names: string[];  // 标签名称
  hit_tags: number[];  // 命中的查询标签索引
  hit_tag_names: string[];  // 命中的查询标签名称
  thumbnail_url?: string;
  hash_code?: number[];
  category_hit: boolean;  // LAll 类别是否命中
  tag_hit: boolean;  // YAll 标签是否命中
}

export interface ImageToTagResult {
  rank: number;
  image_id: number;
  tags: number[];  // YAll 索引
  tag_names: string[];  // 标签名称
  score: number;
  distance: number;
  thumbnail_url?: string;  // 来源图像缩略图URL
}

export interface HitStats {
  total_results: number;
  hits: number;
  hit_rate: number;
  query_tag_count: number;
  query_tags: number[];
  query_tag_names: string[];
  // 两种命中率
  tag_hits: number;
  tag_hit_rate: number;
  category_hits: number;
  category_hit_rate: number;
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
  tag_indices?: number[];
  query_tag_names?: string[];  // 查询标签名称列表
  results: SearchResult[];
  tag_results: ImageToTagResult[];
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
export interface TagStats {
  success: boolean;
  total: number;
  tag_dim: number;
  top_tags: { index: number; count: number }[];
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

// ============== DCMH 检索 API ==============

export interface SearchRequest {
  query_type: 'tag_to_image' | 'image_to_tag' | 'image_to_image';
  tag_indices: number[];
  query_image?: string;  // base64
  dataset: DCMHDataset;
  top_k: number;
  use_encrypted: boolean;
}

export async function search(request: SearchRequest): Promise<UnifiedSearchResponse> {
  const response = await api.post('/api/search', request);
  return response.data;
}

export async function getTagStats(dataset: DCMHDataset = 'flickr25k'): Promise<TagStats> {
  const response = await api.get(`/api/tags/stats?dataset=${dataset}`);
  return response.data;
}

export async function getTagNames(dataset: string): Promise<{ success: boolean; tag_names: string[] }> {
  const response = await api.get(`/api/tags/names/${dataset}`);
  return response.data;
}

// ============== CIR 检索 API ==============

export async function cirSearch(
  queryImage: string,
  dataset: CIRDataset = 'roxford5k',
  topK: number = 10
): Promise<{ success: boolean; results: CIRSearchResult[] }> {
  const response = await api.post('/api/cir/search', {
    query_image: queryImage,
    dataset,
    top_k: topK,
  });
  return response.data;
}

export async function getCIRStatus(dataset: CIRDataset): Promise<CIRStatus> {
  const response = await api.get(`/api/cir/status/${dataset}`);
  return response.data;
}

// ============== 图像 API ==============

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

// ============== 其他 API ==============

export async function rebuildCache(dataset: string = 'flickr25k'): Promise<{ success: boolean; message: string }> {
  const response = await api.post(`/api/search/rebuild-cache?dataset=${dataset}`);
  return response.data;
}

export async function getSearchStatus(dataset: string = 'flickr25k'): Promise<{
  cache_info: any;
  dcmh_status: any;
  aspe_status: any;
  cache_initialized: boolean;
  current_dataset: string;
}> {
  const response = await api.get(`/api/search/status?dataset=${dataset}`);
  return response.data;
}

export default api;