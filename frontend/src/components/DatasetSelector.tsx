'use client';

import { motion } from 'framer-motion';

interface DatasetOption {
  name: string;
  label: string;
}

interface DatasetSelectorProps {
  type: 'dcmh' | 'cir';
  value: string;
  onChange: (dataset: string) => void;
  options?: DatasetOption[];
}

const DEFAULT_DCMH_OPTIONS: DatasetOption[] = [
  { name: 'flickr25k', label: 'Flickr25K' },
  { name: 'nuswide', label: 'NUS-WIDE' },
];

const DEFAULT_CIR_OPTIONS: DatasetOption[] = [
  { name: 'roxford5k', label: 'ROxford5k' },
  { name: 'rparis6k', label: 'RParis6k' },
];

export default function DatasetSelector({
  type,
  value,
  onChange,
  options,
}: DatasetSelectorProps) {
  const datasets = options || (type === 'dcmh' ? DEFAULT_DCMH_OPTIONS : DEFAULT_CIR_OPTIONS);

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-600">数据集：</span>
      <div className="flex space-x-1 p-1 bg-gray-100 rounded-lg">
        {datasets.map((ds) => (
          <button
            key={ds.name}
            onClick={() => onChange(ds.name)}
            className={`relative px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              value === ds.name
                ? 'text-[#6366F1]'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {value === ds.name && (
              <motion.div
                layoutId={`dataset-selector-${type}`}
                className="absolute inset-0 bg-white rounded-md shadow-sm"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.3 }}
              />
            )}
            <span className="relative z-10">{ds.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}