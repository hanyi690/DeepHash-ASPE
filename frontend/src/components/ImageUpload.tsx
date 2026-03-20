'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  onImageUploaded: (imageUrl: string) => void;
}

export default function ImageUpload({ onImageSelected, onImageUploaded }: ImageUploadProps) {
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
        onImageUploaded(data.image_url || previewUrl);
      }
    } catch (error) {
      console.error('上传失败:', error);
    } finally {
      setUploading(false);
    }
  }, [onImageSelected, onImageUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp'],
    },
    maxFiles: 1,
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="text-4xl mb-4">📷</div>
        {isDragActive ? (
          <p className="text-primary-600 font-medium">拖拽图像到此处...</p>
        ) : (
          <>
            <p className="text-gray-700 font-medium mb-1">
              点击或拖拽上传图像
            </p>
            <p className="text-sm text-gray-500">
              支持 JPG、PNG、GIF、WebP 格式
            </p>
          </>
        )}
      </div>

      {preview && (
        <div className="mt-4">
          <div className="relative inline-block">
            <img
              src={preview}
              alt="预览"
              className="max-h-64 rounded-lg shadow-md"
            />
            {uploading && (
              <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
                <div className="text-white">上传中...</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
