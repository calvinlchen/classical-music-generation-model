// frontend/src/App.jsx
import { useState } from "react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

const BACKEND_URL = "http://localhost:8000";

async function playMidiFromArrayBuffer(arrayBuffer) {
  const midi = new Midi(arrayBuffer);

  // master volume a bit lower
  const volume = new Tone.Volume(-6).toDestination();

  const now = Tone.now() + 0.5;

  // one polysynth per track (simple but works)
  midi.tracks.forEach((track) => {
    if (track.notes.length === 0) return;
    const synth = new Tone.PolySynth(Tone.Synth).connect(volume);

    track.notes.forEach((note) => {
      synth.triggerAttackRelease(
        note.name,
        note.duration,
        now + note.time,
        note.velocity
      );
    });
  });
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);

  const handleGenerateAndPlay = async () => {
    try {
      setIsLoading(true);
      setStatus("Starting audio engine…");
      await Tone.start(); // required by browsers

      setStatus("Requesting MIDI from backend…");
      const res = await fetch(`${BACKEND_URL}/generate-midi-from-diffusion`, {
        method: "POST",
      });
      if (!res.ok) {
        throw new Error(`Backend error: ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
      setDownloadUrl(url);

      setStatus("Playing MIDI…");
      const arrayBuffer = await blob.arrayBuffer();
      await playMidiFromArrayBuffer(arrayBuffer);

      setStatus("Done! You can download the file below.");
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#0f172a",
        color: "#e5e7eb",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <div
        style={{
          background: "#020617",
          borderRadius: "1.25rem",
          padding: "2.5rem 2.75rem",
          maxWidth: "640px",
          width: "100%",
          boxShadow: "0 25px 50px -12px rgba(0,0,0,0.65)",
          border: "1px solid rgba(148,163,184,0.25)",
        }}
      >
        <h1
          style={{
            fontSize: "1.75rem",
            fontWeight: 700,
            marginBottom: "0.5rem",
          }}
        >
          "Music" Generator
        </h1>
        <p style={{ marginBottom: "1.5rem", color: "#9ca3af" }}>
          The button below generates a MIDI file from on a diffusion-based
          model. You can also download the generated <code>.mid</code> file.
        </p>

        <button
          onClick={handleGenerateAndPlay}
          disabled={isLoading}
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
            padding: "0.75rem 1.5rem",
            borderRadius: "999px",
            border: "none",
            cursor: isLoading ? "default" : "pointer",
            background: isLoading
              ? "rgba(79,70,229,0.6)"
              : "linear-gradient(135deg,#6366f1,#a855f7)",
            color: "white",
            fontSize: "1rem",
            fontWeight: 600,
            boxShadow:
              "0 10px 25px -12px rgba(88,80,236,0.8),0 0 0 1px rgba(129,140,248,0.4)",
            transition: "transform 0.08s ease, box-shadow 0.08s ease",
          }}
        >
          {isLoading ? "Generating…" : "Generate & Play MIDI"}
        </button>

        <div style={{ marginTop: "1rem", minHeight: "1.5rem" }}>
          {status && (
            <span style={{ fontSize: "0.9rem", color: "#a5b4fc" }}>
              {status}
            </span>
          )}
        </div>

        {downloadUrl && (
          <div style={{ marginTop: "1.5rem" }}>
            <a
              href={downloadUrl}
              download="generated.mid"
              style={{
                color: "#22c55e",
                textDecoration: "none",
                fontWeight: 600,
              }}
            >
              ⬇️ Download generated.mid
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
