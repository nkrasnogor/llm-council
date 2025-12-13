import { useRef } from 'react';
import './FileUploader.css';

export default function FileUploader({ onChange }) {
  const inputRef = useRef(null);

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList || []);
    const readers = files.map(
      (file) =>
        new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () =>
            resolve({
              filename: file.name,
              mime_type: file.type || 'application/octet-stream',
              data_url: reader.result,
            });
          reader.onerror = reject;
          reader.readAsDataURL(file);
        })
    );

    const attachments = await Promise.all(readers);
    onChange(attachments);
  };

  const handleInputChange = (e) => {
    handleFiles(e.target.files);
  };

  const handleClear = () => {
    if (inputRef.current) {
      inputRef.current.value = '';
    }
    onChange([]);
  };

  return (
    <div className="file-uploader">
      <label className="file-label">
        <span>Select files</span>
        <input
          ref={inputRef}
          type="file"
          multiple
          onChange={handleInputChange}
        />
      </label>
      <button type="button" className="clear-files" onClick={handleClear}>
        Clear
      </button>
    </div>
  );
}
