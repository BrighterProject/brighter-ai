"use client";

import {
  Dropzone,
  DropzoneContent,
  DropzoneEmptyState,
} from "../components/kibo-ui/dropzone";

interface DropImageProps {
  onFile: (file: File) => void
}

export const DropImage = ({ onFile }: DropImageProps) => {
  return (
    <Dropzone
      accept={{ "image/jpeg": [], "image/png": [] }}
      className="w-full max-w-md"
      maxFiles={1}
      onDrop={(acceptedFiles) => {
        if (acceptedFiles[0]) onFile(acceptedFiles[0])
      }}
      src={[]}
    >
      <DropzoneEmptyState />
      <DropzoneContent />
    </Dropzone>
  );
};
