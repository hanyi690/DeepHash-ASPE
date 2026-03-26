'use client';

import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TagSelectorProps {
  dataset: 'flickr25k' | 'nuswide';
  selectedTags: number[];
  onChange: (tags: number[]) => void;
  disabled?: boolean;
  placeholder?: string;
}

interface TagInfo {
  index: number;
  name: string;
  count?: number;
}

export default function TagSelector({
  dataset,
  selectedTags,
  onChange,
  disabled = false,
  placeholder = '搜索并选择标签...',
}: TagSelectorProps) {
  const [tagNames, setTagNames] = useState<string[]>([]);
  const [topTags, setTopTags] = useState<{ index: number; count: number }[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  // 加载标签名称
  useEffect(() => {
    const fetchTagNames = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${API_BASE}/api/tags/names/${dataset}`);
        if (response.data.success) {
          setTagNames(response.data.tag_names || []);
        }
      } catch (error) {
        console.error('加载标签名称失败:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTagNames();
  }, [dataset]);

  // 加载常用标签
  useEffect(() => {
    const fetchTopTags = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/tags/stats`);
        if (response.data.success) {
          setTopTags(response.data.top_tags || []);
        }
      } catch (error) {
        console.error('加载常用标签失败:', error);
      }
    };

    fetchTopTags();
  }, []);

  // 转换为带名称的标签信息列表
  const tagInfos = useMemo<TagInfo[]>(() => {
    return tagNames.map((name, index) => ({
      index,
      name,
      count: topTags.find((t) => t.index === index)?.count,
    }));
  }, [tagNames, topTags]);

  // 搜索过滤
  const filteredTags = useMemo<TagInfo[]>(() => {
    if (!searchQuery.trim()) {
      // 默认显示常用标签
      const topIndices = topTags.slice(0, 30).map((t) => t.index);
      return tagInfos.filter((tag) => topIndices.includes(tag.index));
    }

    const query = searchQuery.toLowerCase();
    return tagInfos.filter(
      (tag) =>
        tag.name.toLowerCase().includes(query) ||
        tag.index.toString() === query
    );
  }, [tagInfos, searchQuery, topTags]);

  // 选择/取消选择标签
  const toggleTag = (index: number) => {
    if (disabled) return;

    if (selectedTags.includes(index)) {
      onChange(selectedTags.filter((i) => i !== index));
    } else {
      onChange([...selectedTags, index]);
    }
  };

  // 移除已选标签
  const removeTag = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    onChange(selectedTags.filter((i) => i !== index));
  };

  // 获取已选标签的完整信息
  const selectedTagInfos = useMemo(() => {
    return selectedTags
      .map((index) => ({
        index,
        name: tagNames[index] || `标签 ${index}`,
      }))
      .filter(Boolean);
  }, [selectedTags, tagNames]);

  return (
    <div className="relative">
      {/* 已选标签显示 */}
      {selectedTagInfos.length > 0 && (
        <div className="mb-3 p-3 bg-[#6366F1]/10 rounded-xl">
          <p className="text-sm text-gray-600 mb-2">
            已选择 {selectedTagInfos.length} 个标签：
          </p>
          <div className="flex flex-wrap gap-1.5">
            {selectedTagInfos.map((tag) => (
              <span
                key={tag.index}
                className="inline-flex items-center px-2.5 py-1 bg-white rounded-lg text-sm font-medium text-gray-700 shadow-sm"
              >
                <span className="text-[#6366F1] mr-1">{tag.name}</span>
                <span className="text-gray-400 text-xs">({tag.index})</span>
                <button
                  onClick={(e) => removeTag(tag.index, e)}
                  className="ml-1.5 text-gray-400 hover:text-red-500 transition-colors"
                  disabled={disabled}
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 搜索框 */}
      <div className="relative">
        <div className="relative">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
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
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => setIsOpen(true)}
            placeholder={placeholder}
            disabled={disabled || loading}
            className="input-primary pl-10 pr-4 w-full"
          />
          {loading && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <svg className="animate-spin w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            </div>
          )}
        </div>

        {/* 下拉列表 */}
        <AnimatePresence>
          {isOpen && !disabled && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.15 }}
              className="absolute z-50 mt-2 w-full bg-white rounded-xl shadow-lg border border-gray-200 max-h-80 overflow-hidden"
            >
              {/* 标签列表 */}
              <div className="overflow-y-auto max-h-72 p-2">
                {filteredTags.length > 0 ? (
                  filteredTags.map((tag) => {
                    const isSelected = selectedTags.includes(tag.index);
                    return (
                      <button
                        key={tag.index}
                        onClick={() => toggleTag(tag.index)}
                        className={`w-full text-left px-3 py-2 rounded-lg transition-colors flex items-center justify-between group ${
                          isSelected
                            ? 'bg-[#6366F1]/10 text-[#6366F1]'
                            : 'hover:bg-gray-50 text-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{tag.name}</span>
                          <span className="text-xs text-gray-400">#{tag.index}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {tag.count && (
                            <span className="text-xs text-gray-400">{tag.count} 张</span>
                          )}
                          {isSelected && (
                            <svg className="w-4 h-4 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      </button>
                    );
                  })
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p>未找到匹配的标签</p>
                  </div>
                )}
              </div>

              {/* 底部信息 */}
              <div className="border-t border-gray-100 px-3 py-2 bg-gray-50 text-xs text-gray-500 flex justify-between">
                <span>共 {tagNames.length} 个标签</span>
                <span>已选 {selectedTags.length} 个</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 点击外部关闭下拉框 */}
      {isOpen && (
        <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
      )}
    </div>
  );
}
