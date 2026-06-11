import { DropImage } from "./components/dropzone-images-1"
import { Card } from "./components/ui/card"

const App = () => {
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
        <DropImage />
      </div>
      <Card
        className="mt-4 overflow-hidden"
        style={{ border: "0.5px solid #e2e8f0" }}
      >
        <div
          className="flex items-center px-2 pb-4"
          style={{ borderBottom: "0.5px solid #e2e8f0" }}
        >
          <span className="text-s font-medium text-gray-500">Result</span>
        </div>
        <div className="flex min-h-20 items-center justify-center">
          <p className="text-center text-sm text-gray-400">
            Results will appear here once you upload an image
          </p>
        </div>
      </Card>
    </div>
  )
}

export default App
