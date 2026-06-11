"use client";

import { useState } from "react";
import {
  Dropzone,
  DropzoneContent,
  DropzoneEmptyState,
} from "../components/kibo-ui/dropzone";

export const title = "Images only";

export const DropImage = () => {
  const [files, setFiles] = useState<File[]>([]);

  return (
    <Dropzone
      accept={{ "image/*": [] }}
      className="w-full max-w-md"
      onDrop={(acceptedFiles) => setFiles(acceptedFiles)}
      src={files}
    >
      <DropzoneEmptyState />
      <DropzoneContent />
    </Dropzone>
  );
};
