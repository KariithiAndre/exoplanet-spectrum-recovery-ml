import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { clsx } from 'clsx'

export default function SpectrumUploader({ onUpload, isLoading }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0 && onUpload) {
      onUpload(acceptedFiles[0])
    }
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/fits': ['.fits', '.fit'],
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    disabled: isLoading,
  })

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200',
        isDragActive
          ? 'border-space-400 bg-space-800/50'
          : 'border-space-700 hover:border-space-500 hover:bg-space-800/30',
        isLoading && 'opacity-50 cursor-not-allowed'
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4">
        <div className="w-16 h-16 rounded-full bg-space-800 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-space-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        {isDragActive ? (
          <p className="text-space-300">Drop your spectrum file here...</p>
        ) : (
          <>
            <p className="text-space-300">
              Drag & drop a spectrum file, or click to select
            </p>
            <p className="text-space-500 text-sm">
              Supports FITS, CSV, and JSON formats
            </p>
          </>
        )}
      </div>
    </div>
  )
}
