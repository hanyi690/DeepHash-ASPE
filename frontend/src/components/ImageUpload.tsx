'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  onImageUploaded: (imageUrl: string) => void;
  onImageCleared?: () => void;
}

export default function ImageUpload({ onImageSelected, onImageUploaded, onImageCleared }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    onImageSelected(file);

    // 创建预览
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    // 上传文件
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/images/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // 优先使用本地预览 URL（包含实际图像数据），而不是服务器端点 URL
        // 服务器端点返回的 image_id 是临时生成的，无法用于获取图像
        onImageUploaded(previewUrl);
      }
    } catch (error) {
      console.error('上传失败:', error);
    } finally {
      setUploading(false);
    }
  }, [onImageSelected, onImageUploaded]);

  const handleClear = useCallback(() => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    onImageCleared?.();
  }, [preview, onImageCleared]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp'],
    },
    maxFiles: 1,
  });

  // 有预览图片时只显示预览和取消按钮
  if (preview) {
    return (
      <div className="w-full">
        <div className="relative inline-block group">
          <img
            src={preview}
            alt="预览"
            className="max-h-64 rounded-xl shadow-lg transform group-hover:scale-[1.02] transition-transform duration-300"
          />
          {uploading && (
            <div className="absolute inset-0 bg-black/60 rounded-xl flex items-center justify-center backdrop-blur-sm">
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin mb-2" />
                <span className="text-white font-medium">上传中...</span>
              </div>
            </div>
          )}
          {/* 取消按钮 */}
          {!uploading && (
            <button
              onClick={handleClear}
              className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-lg transition-all duration-200 hover:scale-110"
              title="取消上传"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          {/* 悬停覆盖层 */}
          {!uploading && (
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 rounded-xl transition-all duration-200 pointer-events-none" />
          )}
        </div>
      </div>
    );
  }

  // 没有图片时显示上传区域
  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-all duration-200 overflow-hidden
          ${isDragActive
            ? 'border-[#6366F1] bg-[#6366F1]/5 scale-[1.02]'
            : 'border-gray-300 hover:border-[#6366F1]/60 hover:bg-gray-50'
          }`}
      >
        <input {...getInputProps()} />

        {/* 图标 */}
        <div className="flex justify-center mb-4">
          <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-all duration-200
            ${isDragActive ? 'bg-[#6366F1] text-white' : 'bg-gray-100 text-gray-400'}`}>
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        </div>

        {/* 文本 */}
        {isDragActive ? (
          <div>
            <p className="text-[#6366F1] font-semibold text-lg">拖拽图像到此处...</p>
            <p className="text-gray-500 text-sm mt-1">松开即可上传</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-700 font-medium mb-1">
              点击或拖拽上传图像
            </p>
            <p className="text-sm text-gray-500">
              支持 JPG、PNG、GIF、WebP 格式
            </p>
          </div>
        )}

        {/* 拖拽激活时的装饰 */}
        {isDragActive && (
          <div className="absolute inset-0 border-4 border-[#6366F1]/20 rounded-xl pointer-events-none" />
        )}
      </div>
    </div>
  );
}
