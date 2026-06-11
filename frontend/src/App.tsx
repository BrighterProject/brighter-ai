import { useState } from "react"
import { DropImage } from "./components/dropzone-images-1"
import { Card } from "./components/ui/card"

interface PredictResult {
  room_type: string
  confidence: number
}

type Status = "idle" | "loading" | "success" | "error"

const LABEL: Record<string, string> = {
  bathroom: "Bathroom",
  bedroom: "Bedroom",
  dining: "Dining Room",
  kitchen: "Kitchen",
  livingroom: "Living Room",
}

const App = () => {
  const [status, setStatus] = useState<Status>("idle")
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFile = async (file: File) => {
    setStatus("loading")
    setResult(null)
    setError(null)

    const body = new FormData()
    body.append("image", file)

    try {
      const res = await fetch("/predict", { method: "POST", body })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail ?? `Server error ${res.status}`)
      }
      const data: PredictResult = await res.json()
      setResult(data)
      setStatus("success")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error")
      setStatus("error")
    }
  }

  return (
    <div
      style={{ fontFamily: "Inter, sans-serif" }}
      className="mx-auto max-w-xl px-6 py-8"
    >
      <div className="mb-12 flex items-center gap-2">
        <span
          className="text-sm font-semibold tracking-tight"
          style={{ color: "#0F2D5E" }}
        >
          BrighterAI
        </span>
      </div>

      <div className="mb-6">
        <h1
          className="mb-2 text-2xl font-bold tracking-tight"
          style={{ color: "#0F2D5E" }}
        >
          Room recognition
        </h1>
        <p className="text-sm leading-relaxed text-gray-500">
          Drop a photo of any room and BrighterAI will identify the space.
        </p>
      </div>

      <div
        className="rounded-xl p-8 text-center"
        style={{
          border: "1.5px dashed #4A90D9",
          background: "#EEF4FB",
        }}
      >
        <DropImage onFile={handleFile} />
      </div>

      <Card
        className="mt-4 overflow-hidden"
        style={{ border: "0.5px solid #e2e8f0" }}
      >
        <div
          className="flex items-center px-4 py-3"
          style={{ borderBottom: "0.5px solid #e2e8f0" }}
        >
          <span className="text-s font-medium text-gray-500">Result</span>
        </div>

        <div className="flex min-h-20 items-center justify-center px-4 py-4">
          {status === "idle" && (
            <p className="text-center text-sm text-gray-400">
              Results will appear here once you upload an image
            </p>
          )}

          {status === "loading" && (
            <p className="text-center text-sm text-gray-400">Classifying…</p>
          )}

          {status === "error" && (
            <p className="text-center text-sm text-red-500">{error}</p>
          )}

          {status === "success" && result && (
            <div className="flex w-full items-center justify-between">
              <span
                className="text-base font-semibold"
                style={{ color: "#0F2D5E" }}
              >
                {LABEL[result.room_type] ?? result.room_type}
              </span>
              <span className="text-sm text-gray-400">
                {(result.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

export default App
