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

// ============== 图像 API ==============

export interface ImageUploadResponse {
  success: boolean;
  image_id?: number;
  image_url?: string;
  message: string;
}

export async function uploadImage(file: File): Promise<ImageUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/images/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

export async function extractImageFeature(
  imageId?: number,
  imageData?: string
): Promise<{ success: boolean; hash_code?: number[]; message: string }> {
  const params: any = {};
  if (imageId) params.image_id = imageId;
  if (imageData) params.image_data = imageData;

  const response = await api.post('/api/images/feature', null, { params });
  return response.data;
}

// ============== 文本 API ==============

export interface TextProcessResponse {
  success: boolean;
  text: string;
  hash_code?: number[];
  message: string;
}

export async function processText(text: string): Promise<TextProcessResponse> {
  const response = await api.post('/api/texts/process', { text });
  return response.data;
}

export async function generateTextHash(text: string): Promise<{
  success: boolean;
  hash_code?: number[];
  bit_dim?: number;
  message: string;
}> {
  const response = await api.post('/api/texts/hash', { text });
  return response.data;
}

// ============== 搜索 API ==============

export interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  captions: string[];
  thumbnail_url?: string;
}

export interface SearchResponse {
  success: boolean;
  query_type: string;
  query_text?: string;
  results: SearchResult[];
  total_results: number;
  search_time_ms: number;
  plaintext_map?: number;
  ciphertext_map?: number;
}

export async function search(
  queryType: 'text' | 'image',
  queryText?: string,
  topK: number = 10,
  useEncrypted: boolean = true
): Promise<SearchResponse> {
  const response = await api.post('/api/search', {
    query_type: queryType,
    query_text: queryText,
    top_k: topK,
    use_encrypted: useEncrypted,
  });
  return response.data;
}

export async function demoSearch(
  queryText: string = 'A cat sitting on a chair',
  topK: number = 5
): Promise<SearchResponse> {
  const response = await api.get('/api/search/demo', {
    params: { query_text: queryText, top_k: topK },
  });
  return response.data;
}

// ============== 指标 API ==============

export interface MetricsResponse {
  success: boolean;
  plaintext_i2t_map: number;
  plaintext_t2i_map: number;
  ciphertext_i2t_map: number;
  ciphertext_t2i_map: number;
  i2t_difference: number;
  t2i_difference: number;
  consistent: boolean;
  num_queries: number;
  computation_time_ms: number;
}

export async function getMetrics(
  k?: number,
  numQueries: number = 100
): Promise<MetricsResponse> {
  const response = await api.get('/api/metrics', {
    params: { k, num_queries: numQueries },
  });
  return response.data;
}

export async function getComparison(): Promise<{
  success: boolean;
  bit_dim: number;
  comparison: Array<{
    num_queries: number;
    plaintext_map: number;
    ciphertext_map: number;
    difference: number;
  }>;
}> {
  const response = await api.get('/api/metrics/comparison');
  return response.data;
}

export async function getSystemStatus(): Promise<{
  success: boolean;
  dcmh_status: any;
  aspe_status: any;
  coco_status: any;
}> {
  const response = await api.get('/api/metrics/status');
  return response.data;
}

// ============== 加密 API ==============

export async function encryptDatabase(
  hashCodes: number[][],
  labels?: number[][]
): Promise<{
  success: boolean;
  encrypted_size: number;
  bit_dim: number;
  message: string;
}> {
  const response = await api.post('/api/encrypt/database', {
    hash_codes: hashCodes,
    labels,
  });
  return response.data;
}

export async function generateTrapdoor(hashCode: number[]): Promise<{
  success: boolean;
  encrypted_query?: number[];
  message: string;
}> {
  const response = await api.post('/api/encrypt/trapdoor', {
    hash_code: hashCode,
  });
  return response.data;
}

export async function buildEncryptedDatabase(): Promise<{
  success: boolean;
  num_images: number;
  bit_dim: number;
  message: string;
}> {
  const response = await api.post('/api/encrypt/build-database');
  return response.data;
}

export async function verifyConsistency(numSamples: number = 10): Promise<{
  success: boolean;
  plaintext_map: number;
  ciphertext_map: number;
  difference: number;
  consistent: boolean;
}> {
  const response = await api.post('/api/encrypt/verify-consistency', null, {
    params: { num_samples: numSamples },
  });
  return response.data;
}

// ============== 数据集 API ==============

export async function getDatasetInfo(): Promise<{
  num_images: number;
  num_annotations: number;
  num_unique_captions: number;
  avg_caption_length: number;
  captions_per_image: number;
}> {
  const response = await api.get('/api/metrics/status');
  return response.data.coco_status || {};
}

export async function getImage(imageId: number): Promise<{
  success: boolean;
  image_id: number;
  metadata: any;
  captions: string[];
}> {
  const response = await api.get(`/api/images/${imageId}`);
  return response.data;
}

// ============== CIR 图搜图 API ==============

export interface CIRSearchResult {
  rank: number;
  image_name: string;
  score: number;
}

export interface CIRStatus {
  initialized: boolean;
  indexed: boolean;
  index_size: number;
  feature_dim: number;
  model_loaded: boolean;
  keys_loaded: boolean;
}

export async function getCIRStatus(): Promise<CIRStatus> {
  const response = await api.get('/api/cir/status');
  return response.data;
}

export async function cirSearchUpload(
  file: File,
  topK: number = 10,
  useEncrypted: boolean = true
): Promise<{ results: CIRSearchResult[] }> {
  const formData = new FormData();
  formData.append('image', file);

  const endpoint = useEncrypted
    ? '/api/cir/sknn/search/upload'
    : '/api/cir/search/upload';

  const response = await api.post(endpoint, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    params: { top_k: topK },
  });
  return response.data;
}

// ============== 统一检索 API ==============

export interface UnifiedSearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  labels: number[];
  thumbnail_url?: string;
  hash_code?: number[];
}

export interface ImageToLabelResult {
  rank: number;
  image_id: number;
  labels: number[];
  score: number;
  distance: number;
}

export interface UnifiedSearchResponse {
  success: boolean;
  query_type: string;
  label_indices?: number[];
  results: UnifiedSearchResult[];
  label_results: ImageToLabelResult[];
  total_results: number;
  search_time_ms: number;
  encryption_info?: {
    method: string;
    query_encrypted: boolean;
    database_encrypted: boolean;
    security_level: number;
    bit_dim: number;
  };
  query_hash_code?: number[];
}

export async function unifiedSearch(params: {
  query_type: 'label_to_image' | 'image_to_label' | 'image_to_image';
  label_indices?: number[];
  query_image?: string;
  top_k?: number;
  use_encrypted?: boolean;
}): Promise<UnifiedSearchResponse> {
  const response = await api.post('/api/search', {
    query_type: params.query_type,
    label_indices: params.label_indices,
    query_image: params.query_image,
    top_k: params.top_k || 10,
    use_encrypted: params.use_encrypted ?? true,
  });
  return response.data;
}

// ============== 标签统计 API ==============

export interface LabelStats {
  success: boolean;
  total: number;
  label_dim: number;
  top_labels: { index: number; count: number }[];
}

export async function getLabelStats(): Promise<LabelStats> {
  const response = await api.get('/api/labels/stats');
  return response.data;
}
