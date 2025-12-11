import { useState, useEffect } from "react";
import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

const BACKEND_URL = "http://localhost:8000";

// Hard-coded starting prompts
const PROMPTS = [
  {
    id: "default",
    name: "Default – Mozart, C major, 4/4, 120 BPM",
    text: "<SOS> COMPOSER_mozart KEY_C TIME_SIGNATURE_4/4 TEMPO_BPM_120 MEASURE BEAT",
  },
  {
    id: "blank",
    name: "blank",
    text:"<SOS>"
  },
  {
    id: "haydn_n10",
    name: "Haydn – Piano Piece No. 10",
    text:
      "<SOS> COMPOSER_haydn KEY_G TIME_SIGNATURE_6/8 TEMPO_BPM_120 MEASURE BEAT BEAT BEAT POS_24 NOTE_74 DUR_22 VEL_4 BEAT POS_0 NOTE_55 DUR_20 VEL_3 NOTE_59 DUR_20 VEL_3 NOTE_74 DUR_36 VEL_5 POS_36 NOTE_73 DUR_12 VEL_4 BEAT POS_0 NOTE_55 DUR_10 VEL_3 NOTE_59 DUR_10 VEL_3 NOTE_74 DUR_22 VEL_4 POS_24 NOTE_55 DUR_20 VEL_3 NOTE_59 DUR_20 VEL_3 NOTE_74 DUR_36 VEL_5 BEAT POS_12 NOTE_73 DUR_12 VEL_4 POS_24 NOTE_55 DUR_10 VEL_3 NOTE_59 DUR_10 VEL_3 NOTE_74 DUR_22 VEL_4 MEASURE BEAT POS_0 NOTE_48 DUR_20 VEL_3 NOTE_60 DUR_20 VEL_3 NOTE_76 DUR_36 VEL_5 POS_36 NOTE_75 DUR_12 VEL_4 BEAT POS_0 NOTE_48 DUR_10 VEL_3 NOTE_60 DUR_10 VEL_3 NOTE_76 DUR_22 VEL_4 POS_24 NOTE_50 DUR_20 VEL_3 NOTE_54 DUR_20 VEL_3 NOTE_69 DUR_48 VEL_5 BEAT POS_24 NOTE_50 DUR_10 VEL_3 NOTE_54 DUR_10 VEL_3 NOTE_72 DUR_22 VEL_4 BEAT POS_0 NOTE_55 DUR_20 VEL_3 NOTE_72 DUR_3 VEL_4 POS_3 NOTE_71 DUR_3 VEL_4 POS_6 NOTE_69 DUR_3 VEL_4 POS_9 NOTE_71 DUR_24 VEL_4 POS_36 NOTE_72 DUR_12 VEL_4 BEAT POS_0 NOTE_55 DUR_10 VEL_3 NOTE_74 DUR_22 VEL_4 POS_24 NOTE_48 DUR_20 VEL_3 NOTE_60 DUR_20 VEL_3 NOTE_64 DUR_44 VEL_4 NOTE_71 DUR_3 VEL_4 POS_27 NOTE_69 DUR_3 VEL_4 POS_30 NOTE_68 DUR_3 VEL_4 POS_33 NOTE_69 DUR_24 VEL_4 BEAT POS_12 NOTE_71 DUR_12 VEL_4 POS_24 NOTE_48 DUR_10 VEL_3 NOTE_60 DUR_10 VEL_3 NOTE_64 DUR_22 VEL_4 NOTE_72 DUR_22 VEL_4 MEASURE BEAT",
  },
  {
    id: "mozart_sym28",
    name: "Mozart – Symphony No. 28, Mvt. 4",
    text:
      "<SOS> COMPOSER_mozart KEY_C TIME_SIGNATURE_4/4 TEMPO_BPM_250 MEASURE BEAT POS_0 NOTE_60 DUR_20 VEL_2 NOTE_79 DUR_8 VEL_2 POS_8 NOTE_81 DUR_8 VEL_2 POS_16 NOTE_79 DUR_8 VEL_2 POS_24 NOTE_62 DUR_20 VEL_2 NOTE_77 DUR_12 VEL_2 POS_36 NOTE_79 DUR_12 VEL_2 BEAT POS_0 NOTE_64 DUR_20 VEL_2 POS_24 NOTE_60 DUR_20 VEL_2 BEAT POS_0 NOTE_62 DUR_20 VEL_2 NOTE_77 DUR_8 VEL_2 POS_8 NOTE_79 DUR_8 VEL_2 POS_16 NOTE_77 DUR_8 VEL_2 POS_24 NOTE_64 DUR_20 VEL_2 NOTE_76 DUR_12 VEL_2 POS_36 NOTE_77 DUR_12 VEL_2 BEAT POS_0 NOTE_65 DUR_20 VEL_2 POS_24 NOTE_62 DUR_20 VEL_2 MEASURE BEAT POS_0 NOTE_64 DUR_20 VEL_2 NOTE_76 DUR_8 VEL_2 POS_8 NOTE_77 DUR_8 VEL_2 POS_16 NOTE_76 DUR_8 VEL_2 POS_24 NOTE_65 DUR_20 VEL_2 NOTE_74 DUR_12 VEL_2 POS_36 NOTE_76 DUR_12 VEL_2 BEAT POS_0 NOTE_67 DUR_20 VEL_2 POS_24 NOTE_64 DUR_20 VEL_2 BEAT POS_0 NOTE_65 DUR_20 VEL_2 NOTE_74 DUR_8 VEL_2 POS_8 NOTE_76 DUR_8 VEL_2 POS_16 NOTE_74 DUR_8 VEL_2 POS_24 NOTE_67 DUR_20 VEL_2 NOTE_72 DUR_12 VEL_2 POS_36 NOTE_74 DUR_12 VEL_2 BEAT POS_0 NOTE_69 DUR_20 VEL_2 POS_24 NOTE_65 DUR_20 VEL_2 MEASURE BEAT POS_0 NOTE_64 DUR_20 VEL_2 NOTE_72 DUR_8 VEL_2 POS_8 NOTE_74 DUR_8 VEL_2 POS_16 NOTE_72 DUR_8 VEL_2 POS_24 NOTE_65 DUR_20 VEL_2 NOTE_71 DUR_12 VEL_2 POS_36 NOTE_72 DUR_12 VEL_2 BEAT POS_0 NOTE_67 DUR_20 VEL_2 POS_24 NOTE_64 DUR_20 VEL_2 BEAT POS_0 NOTE_62 DUR_20 VEL_2 NOTE_71 DUR_8 VEL_2 POS_8 NOTE_72 DUR_8 VEL_2 POS_16 NOTE_71 DUR_8 VEL_2 POS_24 NOTE_64 DUR_20 VEL_2 NOTE_69 DUR_12 VEL_2 POS_36 NOTE_71 DUR_12 VEL_2 BEAT POS_0 NOTE_65 DUR_20 VEL_2 POS_24 NOTE_62 DUR_20 VEL_2 MEASURE BEAT",
  },
];

const COMPOSER_OPTIONS = [
  { value: "mozart", label: "Mozart" },
  { value: "haydn", label: "Haydn" },
  { value: "beethoven", label: "Beethoven" },
];

const KEY_OPTIONS = [
  "C", "Cm", "Db", "C#m", "D", "Dm", "Eb", "Ebm", "E", "Em", 
  "F", "Fm", "Gb", "F#m", "G", "Gm", "Ab", "G#m", "A", "Am", 
  "Bb", "Bbm", "B", "Bm"
];

// ----------------- shared helpers -----------------

function base64ToUint8Array(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

let currentSynth = null;

function stopToneTransportAndSynth() {
  Tone.Transport.stop();
  Tone.Transport.cancel();
  if (currentSynth) {
    currentSynth.dispose();
    currentSynth = null;
  }
}

/**
 * Play a Tone.js Midi object starting at a given position in seconds.
 * Rebuilds the schedule each time (so seeking backwards works).
 */
async function playMidiFromPosition(midi, startSec = 0) {
  await Tone.start();

  // Reset everything
  stopToneTransportAndSynth();

  const volume = new Tone.Volume(-6).toDestination();
  currentSynth = new Tone.PolySynth(Tone.Synth).connect(volume);

  midi.tracks.forEach((track) => {
    track.notes.forEach((note) => {
      const noteStart = note.time;
      const noteEnd = note.time + note.duration;

      // Skip notes that have already finished before startSec
      if (noteEnd <= startSec) return;

      const scheduledTime = Math.max(0, noteStart - startSec);

      Tone.Transport.schedule((time) => {
        currentSynth.triggerAttackRelease(
          note.name,
          note.duration,
          time,
          note.velocity
        );
      }, scheduledTime);
    });
  });

  Tone.Transport.seconds = 0;
  Tone.Transport.start("+0.05");
}

// ----------------- playback hook -----------------

function useMidiPlayer(midiObj) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0); // 0–1
  const [duration, setDuration] = useState(0); // seconds
  const [offset, setOffset] = useState(0);     // where playback starts in seconds

  // When a new MIDI object arrives, reset state
  useEffect(() => {
    if (midiObj) {
      const dur = midiObj.duration || 0;
      setDuration(dur);
      setProgress(0);
      setOffset(0);
      setIsPlaying(false);
    } else {
      setDuration(0);
      setProgress(0);
      setOffset(0);
      setIsPlaying(false);
    }
  }, [midiObj]);

  // Progress loop
  useEffect(() => {
    let rafId;
    const update = () => {
      if (!isPlaying || duration <= 0) return;

      const t = offset + Tone.Transport.seconds;
      const frac = Math.min(t / duration, 1);
      setProgress(frac);

      if (t >= duration) {
        stopToneTransportAndSynth();
        setIsPlaying(false);
        setProgress(1);
        return;
      }
      rafId = requestAnimationFrame(update);
    };

    if (isPlaying && duration > 0) {
      rafId = requestAnimationFrame(update);
    }

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [isPlaying, duration, offset]);

  // Start/resume playback from the current offset
  const play = async (overrideMidiObj) => {
    const midiToPlay = overrideMidiObj || midiObj;
    if (!midiToPlay) return;
    await playMidiFromPosition(midiToPlay, offset);
    setIsPlaying(true);
  };

  const stop = () => {
    stopToneTransportAndSynth();
    setIsPlaying(false);
    // keep offset & progress where they are so you can resume
  };

  // Seek to a new fraction of the piece (0–1)
  const seek = async (fraction) => {
    if (!midiObj || duration <= 0) {
      setProgress(fraction);
      return;
    }

    const clamped = Math.min(Math.max(fraction, 0), 1);
    const newPosSec = clamped * duration;

    setOffset(newPosSec);
    setProgress(clamped);

    // If currently playing, re-start from the new position
    if (isPlaying) {
      await playMidiFromPosition(midiObj, newPosSec);
      setIsPlaying(true);
    }
  };

  return {
    isPlaying,
    progress,
    duration,
    play,
    stop,
    seek,
  };
}

// ----------------- Diffusion page -----------------

function DiffusionPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [midiObj, setMidiObj] = useState(null);

  // --- New State for Conditional Inputs ---
  const [mode, setMode] = useState("unconditional"); // "unconditional" | "conditional"
  const [composer, setComposer] = useState("mozart");
  const [selectedKey, setSelectedKey] = useState("C");
  const [bpm, setBpm] = useState(120);
  const [guidance, setGuidance] = useState(7.0);

  const { isPlaying, progress, duration, play, stop, seek } = useMidiPlayer(midiObj);

  // cleanup download URL & audio when unmount
  useEffect(() => {
    return () => {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      stopToneTransportAndSynth();
    };
  }, [downloadUrl]);

  const handleGenerate = async () => {
    try {
      setIsLoading(true);
      
      let endpoint = "/generate-midi-from-diffusion";
      let body = null;
      let headers = {};

      if (mode === "conditional") {
        setStatus(`Requesting ${composer} in ${selectedKey} at ${bpm} BPM...`);
        endpoint = "/generate-midi-from-diffusion-conditional";
        headers = { "Content-Type": "application/json" };
        body = JSON.stringify({
          composer: composer,
          key: selectedKey,
          bpm: Number(bpm),
          guidance: Number(guidance)
        });
      } else {
        setStatus("Requesting unconditional random sample...");
      }

      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: "POST",
        headers: headers,
        body: body
      });

      if (!res.ok) {
        let extra = "";
        try {
          const text = await res.text();
          extra = ` – ${text}`;
        } catch (_) {}
        throw new Error(`Backend error: ${res.status}${extra}`);
      }

      const data = await res.json();

      // MIDI
      const midiBytes = base64ToUint8Array(data.midi_base64);
      const midiBlob = new Blob([midiBytes], { type: "audio/midi" });
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      const url = URL.createObjectURL(midiBlob);
      setDownloadUrl(url);

      const arrayBuf = await midiBlob.arrayBuffer();
      const midi = new Midi(arrayBuf);
      setMidiObj(midi);

      // Image
      setImageUrl(`data:image/png;base64,${data.image_base64}`);

      setStatus("Generation successful. Ready to play.");
      
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const labelStyle = { fontSize: "0.85rem", color: "#cbd5e1", display: "block", marginBottom: "0.3rem" };
  const inputStyle = {
    width: "100%",
    padding: "0.4rem",
    borderRadius: "0.3rem",
    background: "#1e293b",
    color: "#e5e7eb",
    border: "1px solid #475569",
    marginBottom: "0.8rem"
  };

  return (
    <>
      <h1
        style={{
          fontSize: "1.75rem",
          fontWeight: 700,
          marginBottom: "0.5rem",
        }}
      >
        Diffusion-Based MIDI Generator
      </h1>
      <p style={{ marginBottom: "1.5rem", color: "#9ca3af" }}>
        Generate piano rolls using one of two diffusion models, trained in-house.<br></br><br></br>
        Choose "Unconditional" to generate a random new piano-roll image, which will
        be converted into MIDI audio.<br></br><br></br>
        Choose "Conditional" to generate from a model which was trained using
        classifer-free guidance. In this mode, you can control basic style parameters.
      </p>

      {/* --- Mode Toggle --- */}
      <div style={{ 
        display: "flex", 
        gap: "1rem", 
        marginBottom: "1.5rem", 
        background: "rgba(30, 41, 59, 0.5)", 
        padding: "0.5rem", 
        borderRadius: "0.5rem",
        width: "fit-content"
      }}>
        <label style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: "0.4rem", color: "#e5e7eb" }}>
          <input 
            type="radio" 
            name="diffMode" 
            value="unconditional" 
            checked={mode === "unconditional"} 
            onChange={() => setMode("unconditional")}
          />
          Unconditional (Random)
        </label>
        <label style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: "0.4rem", color: "#e5e7eb" }}>
          <input 
            type="radio" 
            name="diffMode" 
            value="conditional" 
            checked={mode === "conditional"} 
            onChange={() => setMode("conditional")}
          />
          Conditional (Style)
        </label>
      </div>

      {/* --- Conditional Controls Area --- */}
      {mode === "conditional" && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "1rem",
          background: "rgba(99, 102, 241, 0.1)",
          border: "1px solid rgba(99, 102, 241, 0.3)",
          borderRadius: "0.75rem",
          padding: "1rem",
          marginBottom: "1.5rem"
        }}>
          <div>
            <label style={labelStyle}>Composer</label>
            <select style={inputStyle} value={composer} onChange={(e) => setComposer(e.target.value)}>
              {COMPOSER_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={labelStyle}>Key</label>
            <select style={inputStyle} value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
              {KEY_OPTIONS.map(k => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={labelStyle}>BPM</label>
            <input 
              type="number" 
              min="40" 
              max="240" 
              style={inputStyle} 
              value={bpm} 
              onChange={(e) => setBpm(e.target.value)} 
            />
          </div>

          <div>
            <label style={labelStyle}>
              Guidance Scale ({guidance})
              <span style={{ fontSize: "0.7em", color: "#94a3b8", marginLeft: "5px" }}>
                Higher = stricter adherence to given parameters
              </span>
            </label>
            <input 
              type="range" 
              min="1" 
              max="10" 
              step="0.5" 
              style={{ width: "100%", cursor: "pointer" }} 
              value={guidance} 
              onChange={(e) => setGuidance(e.target.value)} 
            />
          </div>
        </div>
      )}

      {/* --- Generate Button --- */}
      <button
        onClick={handleGenerate}
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
        {isLoading ? "Generating..." : `Generate (${mode})`}
      </button>

      <div style={{ marginTop: "1rem", minHeight: "1.5rem" }}>
        {status && (
          <span style={{ fontSize: "0.9rem", color: "#a5b4fc" }}>{status}</span>
        )}
      </div>

      {/* --- Playback & Download Section (Identical to previous) --- */}
      {midiObj && (
        <div
          style={{
            marginTop: "1.75rem",
            padding: "1rem 1.25rem",
            borderRadius: "0.75rem",
            background: "rgba(15,23,42,0.9)",
            border: "1px solid rgba(148,163,184,0.35)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: "1rem",
              marginBottom: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <div style={{ fontSize: "0.9rem", color: "#cbd5f5" }}>
              Progress: <strong>{Math.round(progress * 100)}%</strong>
            </div>
            <div style={{ fontSize: "0.9rem", color: "#94a3b8" }}>
              Length: {duration.toFixed(2)} s
            </div>
          </div>

          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(progress * 100)}
            onChange={(e) => {
              const frac = Number(e.target.value) / 100;
              seek(frac);
            }}
            style={{ width: "100%" }}
          />

          <div
            style={{
              marginTop: "0.9rem",
              display: "flex",
              gap: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={() => play()}
              style={{
                padding: "0.5rem 1.2rem",
                borderRadius: "999px",
                border: "none",
                cursor: "pointer",
                background: "rgba(52,211,153,0.16)",
                color: "#6ee7b7",
                fontWeight: 600,
              }}
            >
              ▶ Play
            </button>
            <button
              onClick={stop}
              disabled={!isPlaying}
              style={{
                padding: "0.5rem 1.2rem",
                borderRadius: "999px",
                border: "none",
                cursor: isPlaying ? "pointer" : "default",
                background: "rgba(248,113,113,0.16)",
                color: "#fca5a5",
                fontWeight: 600,
              }}
            >
              ⏹ Stop
            </button>
          </div>
        </div>
      )}

      {downloadUrl && (
        <div style={{ marginTop: "1.5rem" }}>
          <a
            href={downloadUrl}
            download="generated_diffusion.mid"
            style={{
              color: "#22c55e",
              textDecoration: "none",
              fontWeight: 600,
            }}
          >
            ⬇️ Download diffusion-based MIDI
          </a>
        </div>
      )}

      {imageUrl && (
        <div style={{ marginTop: "2rem" }}>
          <h2
            style={{
              fontSize: "1.1rem",
              fontWeight: 600,
              marginBottom: "0.75rem",
            }}
          >
            Generated piano-roll (inverted y-axis)
          </h2>
          <div
            style={{
              borderRadius: "0.75rem",
              overflow: "hidden",
              border: "1px solid rgba(148,163,184,0.35)",
              background: "black",
            }}
          >
            <img
              src={imageUrl}
              alt="Generated piano-roll"
              style={{
                display: "block",
                width: "100%",
                height: "auto",
              }}
            />
          </div>
        </div>
      )}
    </>
  );
}

// ----------------- Transformer page (Merged with Smart AI) -----------------

function TransformerPage() {
  // --- 1. NEW STATE VARIABLES (Make sure these are here!) ---
  // We initialize the prompt using the first template in your PROMPTS list
  const [prompt, setPrompt] = useState(PROMPTS[0].text);
  const [selectedId, setSelectedId] = useState(PROMPTS[0].id);
  const [modelType, setModelType] = useState("inhouse");

  const [maxTokens, setMaxTokens] = useState(500);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [midiObj, setMidiObj] = useState(null);
  const [generatedText, setGeneratedText] = useState("");

  // --- AI Assistant State ---
  const [aiDescription, setAiDescription] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiDraft, setAiDraft] = useState(null);

  const promptTokenCount = prompt.trim() ? prompt.trim().split(/\s+/).length : 0;
  const maxNewTokensNumber = Number(maxTokens) || 500;
  const totalTokensForGPT2 = promptTokenCount + maxNewTokensNumber;
  const gpt2TokensRemaining = 1024 - promptTokenCount;
  const exceedsGpt2Limit = modelType === "gpt2" && totalTokensForGPT2 > 1024;

  const { isPlaying, progress, duration, play, stop, seek } = useMidiPlayer(midiObj);

  // Cleanup
  useEffect(() => {
    return () => {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      stopToneTransportAndSynth();
    };
  }, [downloadUrl]);

  // --- 2. NEW HANDLER FUNCTION (Make sure this is here!) ---
  const handleTemplateChange = (e) => {
    const newId = e.target.value;
    setSelectedId(newId);
    
    // Find the text associated with this ID and update the main prompt
    const template = PROMPTS.find(p => p.id === newId);
    if (template) {
      setPrompt(template.text);
    }
  };

  const handleDraftTokens = async () => {
    if (!aiDescription.trim()) return;
    setAiLoading(true);
    setAiDraft(null); 

    try {
      const res = await fetch(`${BACKEND_URL}/generate-seed-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_prompt: aiDescription }),
      });

      if (!res.ok) {
        let extra = "";
        try {
          const text = await res.text();
          extra = ` – ${text}`;
        } catch (_) {}
        throw new Error(`Backend error: ${res.status}${extra}`);
      }
      const data = await res.json();
      setAiDraft(data.seed_text); 
    } catch (err) {
      console.error(err);
      setStatus(`AI Error: ${err.message}`);
    } finally {
      setAiLoading(false);
    }
  };

  const handleGenerate = async () => {
    const tokensForRequest = maxNewTokensNumber;
    setGeneratedText("");

    if (modelType === "gpt2" && exceedsGpt2Limit) {
      const allowed = Math.max(1024 - promptTokenCount, 0);
      setStatus(
        `GPT-2 limit exceeded: prompt has ${promptTokenCount} tokens. ` +
        `Reduce max new tokens to ${allowed} or shorten the prompt.`
      );
      return;
    }

    try {
      setIsLoading(true);
      setStatus(
        modelType === "gpt2"
          ? "Generating with GPT-2 tuned model..."
          : "Generating music from tokens..."
      );

      const res = await fetch(`${BACKEND_URL}/generate-midi-from-transformer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_text: prompt,
          max_new_tokens: tokensForRequest,
          model_type: modelType,
        }),
      });

      if (!res.ok) {
        let extra = "";
        try {
          const text = await res.text();
          extra = ` – ${text}`;
        } catch (_) {}
        throw new Error(`Backend error: ${res.status}${extra}`);
      }
      const data = await res.json();

      const midiBytes = base64ToUint8Array(data.midi_base64);
      const midiBlob = new Blob([midiBytes], { type: "audio/midi" });
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      const url = URL.createObjectURL(midiBlob);
      setDownloadUrl(url);

      const arrayBuf = await midiBlob.arrayBuffer();
      const midi = new Midi(arrayBuf);
      setMidiObj(midi);
      setGeneratedText(data.generated_text || "");

      setStatus("Generation complete.");
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <h1 style={{ fontSize: "1.75rem", fontWeight: 700, marginBottom: "0.5rem" }}>
        Transformer-Based MIDI Generator
      </h1>
      <p style={{ marginBottom: "1.5rem", color: "#9ca3af" }}>
        Generate MIDI tokens with the text transformer. Use the AI assistant, or pick a starting template below.
      </p>

      {/* --- MODEL SELECT --- */}
      <div style={{
        display: "grid",
        gap: "0.35rem",
        background: "rgba(15,23,42,0.7)",
        border: "1px solid rgba(148,163,184,0.3)",
        borderRadius: "0.75rem",
        padding: "0.75rem",
        marginBottom: "1.25rem"
      }}>
        <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "#e5e7eb" }}>
          Choose model
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#cbd5e1", cursor: "pointer" }}>
          <input
            type="radio"
            name="modelType"
            value="gpt2"
            checked={modelType === "gpt2"}
            onChange={() => setModelType("gpt2")}
          />
          GPT-2 tuned (1024-token context window)
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#cbd5e1", cursor: "pointer" }}>
          <input
            type="radio"
            name="modelType"
            value="inhouse"
            checked={modelType === "inhouse"}
            onChange={() => setModelType("inhouse")}
          />
          In-house transformer (no token limit)
        </label>
      </div>

      {/* --- AI ASSISTANT SECTION --- */}
      <div style={{
        background: "rgba(99, 102, 241, 0.1)",
        border: "1px dashed rgba(99, 102, 241, 0.4)",
        borderRadius: "0.75rem",
        padding: "1rem",
        marginBottom: "2rem"
      }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "#818cf8", marginTop: 0 }}>
          AI Prompt Assistant
        </h3>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <input
            type="text"
            placeholder="Describe the basics (e.g. 'Fast piece in A major')..."
            value={aiDescription}
            onChange={(e) => setAiDescription(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleDraftTokens()}
            style={{
              flex: 1,
              minWidth: "200px",
              padding: "0.5rem",
              borderRadius: "0.5rem",
              border: "1px solid rgba(148,163,184,0.4)",
              background: "#020617",
              color: "#e5e7eb",
            }}
          />
          <button
            onClick={handleDraftTokens}
            disabled={aiLoading || !aiDescription}
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "0.5rem",
              border: "none",
              cursor: aiLoading ? "wait" : "pointer",
              background: "#4f46e5",
              color: "white",
              fontWeight: 600,
            }}
          >
            {aiLoading ? "Drafting..." : "Draft Tokens"}
          </button>
        </div>

        {aiDraft && (
          <div style={{ marginTop: "1rem", background: "rgba(0,0,0,0.3)", padding: "0.75rem", borderRadius: "0.5rem" }}>
            <div style={{ fontSize: "0.8rem", color: "#cbd5e1", marginBottom: "0.5rem" }}>
              ChatGPT suggested:
            </div>
            <div style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "#a5b4fc", marginBottom: "0.75rem", wordBreak: "break-all" }}>
              {aiDraft}
            </div>
            <button
              onClick={() => {
                setPrompt(aiDraft);
                // We use 'blank' here because that's the ID in your PROMPTS array
                setSelectedId("blank"); 
              }}
              style={{
                fontSize: "0.8rem",
                padding: "0.3rem 0.8rem",
                background: "#4338ca",
                color: "white",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer"
              }}
            >
              ⬇️ Paste into Main Input
            </button>
          </div>
        )}
      </div>

      {/* --- 3. TEMPLATE DROPDOWN --- */}
      <div style={{ marginBottom: "0.5rem" }}>
        <label style={{ fontSize: "0.9rem", color: "#cbd5e1" }}>
          Starting template:&nbsp;
          <select 
            value={selectedId} 
            onChange={handleTemplateChange}
            style={{
              marginLeft: "0.5rem",
              padding: "0.3rem",
              borderRadius: "0.3rem",
              background: "#1e293b",
              color: "#e5e7eb",
              border: "1px solid #475569"
            }}
          >
            {PROMPTS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* --- MAIN INPUT AREA --- */}
      <label style={{ fontSize: "0.9rem", color: "#cbd5e1" }}>
        <strong>Main Token Input</strong> (The model will continue from here):
        <textarea
          value={prompt}
          onChange={(e) => {
            setPrompt(e.target.value);
            // Switch dropdown to 'blank' if user types manually
            if (selectedId !== 'blank') setSelectedId('blank');
          }}
          rows={4}
          style={{
            marginTop: "0.4rem",
            width: "100%",
            resize: "vertical",
            borderRadius: "0.5rem",
            border: "1px solid rgba(148,163,184,0.4)",
            background: "#020617",
            color: "#e5e7eb",
            padding: "0.5rem 0.75rem",
            fontFamily: "monospace",
            lineHeight: "1.4"
          }}
        />
      </label>

       <div style={{ marginTop: "0.75rem", marginBottom: "1.25rem" }}>
        <label style={{ fontSize: "0.9rem", color: "#cbd5e1" }}>
          Max new tokens:&nbsp;
          <input
            type="number"
            min={modelType === "gpt2" ? 0 : 10}
            max={modelType === "gpt2" ? Math.max(gpt2TokensRemaining, 0) : 2048}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{
              width: "90px",
              borderRadius: "0.5rem",
              border: "1px solid rgba(148,163,184,0.4)",
              background: "#020617",
              color: "#e5e7eb",
              padding: "0.25rem 0.5rem",
            }}
          />
        </label>
      </div>

      <button
        onClick={handleGenerate}
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
        }}
      >
        {isLoading ? "Generating..." : "Generate MIDI"}
      </button>

      <div style={{ marginTop: "1rem", minHeight: "1.5rem" }}>
        {status && (
          <span style={{ fontSize: "0.9rem", color: "#a5b4fc" }}>{status}</span>
        )}
      </div>

      {midiObj && (
        <div
          style={{
            marginTop: "1.75rem",
            padding: "1rem 1.25rem",
            borderRadius: "0.75rem",
            background: "rgba(15,23,42,0.9)",
            border: "1px solid rgba(148,163,184,0.35)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: "1rem",
              marginBottom: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <div style={{ fontSize: "0.9rem", color: "#cbd5f5" }}>
              Progress: <strong>{Math.round(progress * 100)}%</strong>
            </div>
            <div style={{ fontSize: "0.9rem", color: "#94a3b8" }}>
              Length: {duration.toFixed(2)} s
            </div>
          </div>

          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(progress * 100)}
            onChange={(e) => {
              const frac = Number(e.target.value) / 100;
              seek(frac);
            }}
            style={{ width: "100%", cursor: "pointer" }}
          />

          <div
            style={{
              marginTop: "0.9rem",
              display: "flex",
              gap: "0.75rem",
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={() => play()}
              style={{
                padding: "0.5rem 1.2rem",
                borderRadius: "999px",
                border: "none",
                cursor: "pointer",
                background: "rgba(52,211,153,0.16)",
                color: "#6ee7b7",
                fontWeight: 600,
              }}
            >
              ▶ Play
            </button>
            <button
              onClick={stop}
              disabled={!isPlaying}
              style={{
                padding: "0.5rem 1.2rem",
                borderRadius: "999px",
                border: "none",
                cursor: isPlaying ? "pointer" : "default",
                background: "rgba(248,113,113,0.16)",
                color: "#fca5a5",
                fontWeight: 600,
              }}
            >
              ⏹ Stop
            </button>
          </div>
        </div>
      )}

      {downloadUrl && (
        <div style={{ marginTop: "1.5rem" }}>
          <a
            href={downloadUrl}
            download="transformer_generated.mid"
            style={{
              color: "#22c55e",
              textDecoration: "none",
              fontWeight: 600,
            }}
          >
            Download MIDI
          </a>
        </div>
      )}

      {generatedText && (
        <div
          style={{
            marginTop: "0.75rem",
            background: "rgba(15,23,42,0.7)",
            border: "1px solid rgba(148,163,184,0.35)",
            borderRadius: "0.5rem",
            padding: "0.75rem",
            maxHeight: "180px",
            overflowY: "auto",
          }}
        >
          <div
            style={{
              fontSize: "0.9rem",
              color: "#cbd5e1",
              marginBottom: "0.35rem",
            }}
          >
            Generated token sequence
          </div>
          <div
            style={{
              fontFamily: "monospace",
              fontSize: "0.85rem",
              color: "#a5b4fc",
              wordBreak: "break-word",
              whiteSpace: "pre-wrap",
            }}
          >
            {generatedText}
          </div>
        </div>
      )}
    </>
  );
}

// ----------------- About page -----------------

function AboutPage() {
  const sectionStyle = {
    background: "rgba(15,23,42,0.85)",
    border: "1px solid rgba(148,163,184,0.25)",
    borderRadius: "1rem",
    padding: "1.25rem",
    marginBottom: "1rem",
  };

  const headingStyle = {
    margin: "0 0 0.5rem 0",
    fontSize: "1.05rem",
    color: "#e5e7eb",
  };

  const bodyStyle = { color: "#cbd5e1", fontSize: "0.95rem", lineHeight: 1.6 };

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <div style={sectionStyle}>
        <h2 style={headingStyle}>Project Purpose</h2>
        <p style={bodyStyle}>
          This web app serves as an experiment to make classical-style music generation using accessible machine
          learning methods. Specifically, we train on a dataset of only classical-era (roughly 1725-1800 A.D.) composers --
          Haydn, Mozart, and Beethoven -- in an attempt to create models with a roughly consistent tonal style. While this
          is thus limited also by 
        </p>
      </div>

      <div style={sectionStyle}>
        <h2 style={headingStyle}>Training Data & Conversion</h2>
        <p style={bodyStyle}>
          Source data: curated MIDI scores from Haydn, Mozart, and Beethoven. MIDI files collected from a public dataset:
          <a href="https://huggingface.co/datasets/drengskapur/midi-classical-music"> huggingface.co/datasets/drengskapur/midi-classical-music</a> <br></br><br></br>
          Each file is normalized (tempo/key/time-signature) and converted to two parallel representations:
        </p>
        <ul style={{ ...bodyStyle, paddingLeft: "1.2rem", margin: "0.35rem 0" }}>
          <li>
            <strong>Text tokens:</strong> metadata tokens (COMPOSER_x, KEY_x, TIME_SIGNATURE_a/b, TEMPO_BPM_x),
            structural tokens (&lt;SOS&gt;, MEASURE, BEAT, POS_i), and note tokens (NOTE_p, DUR_t in grid units,
            VEL_v binned). This stream feeds the transformer models.
          </li>
          <li>
            <strong>Piano-roll images:</strong> 88-by-1024 grids (pitch vs. time) used as diffusion targets; note density
            and rhythm patterns are learned directly from these images.
          </li>
        </ul>
      </div>

      <div style={sectionStyle}>
        <h2 style={headingStyle}>Transformer Models</h2>
        <ul style={{ ...bodyStyle, paddingLeft: "1.2rem", margin: 0 }}>
          <li>
            <strong>In-house transformer:</strong> custom causal encoder (512-d hidden, 8 layers, 8 heads) trained from scratch
            on the tokenized classical corpus. No UI-enforced context limit.
          </li>
          <li>
            <strong>GPT-2 tuned:</strong> pretrained GPT-2 LM head fine-tuned on the same tokenization; 1024-token context window
            (prompt + generated). The frontend enforces this window when GPT-2 mode is selected.
          </li>
        </ul>
      </div>

      <div style={sectionStyle}>
        <h2 style={headingStyle}>Diffusion Models</h2>
        <ul style={{ ...bodyStyle, paddingLeft: "1.2rem", margin: 0 }}>
          <li>
            <strong>Unconditional diffusion:</strong> UNet denoises piano-roll images from pure noise to produce free-form samples.
          </li>
          <li>
            <strong>Conditional CFG diffusion:</strong> UNet with composer/key/tempo embeddings; classifier-free guidance controls
            how strongly the chosen conditions influence the output.
          </li>
        </ul>
      </div>
    </div>
  );
}

// ----------------- Root app with tabs -----------------

function App() {
  // Options: "transformer" | "diffusion" | "about"
  const [tab, setTab] = useState("transformer"); 

  const getButtonStyle = (active) => ({
    padding: "0.4rem 0.9rem",
    borderRadius: "999px",
    border: "none",
    cursor: "pointer",
    fontSize: "0.9rem",
    fontWeight: 600,
    background: active ? "#1d4ed8" : "rgba(15,23,42,0.9)",
    color: active ? "#e5e7eb" : "#9ca3af",
    transition: "all 0.2s ease"
  });

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
          maxWidth: "960px",
          width: "100%",
          boxShadow: "0 25px 50px -12px rgba(0,0,0,0.65)",
          border: "1px solid rgba(148,163,184,0.25)",
        }}
      >
        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: "0.75rem",
            marginBottom: "1.75rem",
            flexWrap: "wrap"
          }}
        >
          <button
            onClick={() => setTab("transformer")}
            style={getButtonStyle(tab === "transformer")}
          >
            Transformer
          </button>
          <button
            onClick={() => setTab("diffusion")}
            style={getButtonStyle(tab === "diffusion")}
          >
            Diffusion
          </button>
          <button
            onClick={() => setTab("about")}
            style={getButtonStyle(tab === "about")}
          >
            About
          </button>
        </div>

        {/* Page Routing */}
        {tab === "diffusion" && <DiffusionPage />}
        {tab === "transformer" && <TransformerPage />}
        {tab === "about" && <AboutPage />}
      </div>
    </div>
  );
}

export default App;
