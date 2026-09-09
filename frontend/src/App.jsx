import { useEffect, useRef, useState } from "react"
import "./App.css"

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {

  // =========================================
  // MAIN STATE
  // =========================================

  const [audioFile, setAudioFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)

  const [isRecording, setIsRecording] =
    useState(false)

  const [recordingTime, setRecordingTime] =
    useState(0)

  const [isAnalyzing, setIsAnalyzing] =
    useState(false)

  const [result, setResult] =
    useState(null)

  const [error, setError] =
    useState("")

  const [verificationStatus, setVerificationStatus] =
    useState(null)

  const [demoMode, setDemoMode] =
    useState(false)


  // =========================================
  // ANALYSIS HISTORY
  // =========================================

  const [history, setHistory] = useState(() => {

    try {

      const savedHistory =
        localStorage.getItem(
          "nexashield_analysis_history"
        )

      return savedHistory
        ? JSON.parse(savedHistory)
        : []

    } catch {

      return []

    }

  })


  // =========================================
  // REFS
  // =========================================

  const mediaRecorderRef =
    useRef(null)

  const timerRef =
    useRef(null)

  const chunksRef =
    useRef([])


  // =========================================
  // SAVE HISTORY
  // =========================================

  useEffect(() => {

    localStorage.setItem(
      "nexashield_analysis_history",
      JSON.stringify(history)
    )

  }, [history])


  // =========================================
  // ADD ANALYSIS TO HISTORY
  // =========================================

  const addToHistory = (
    data,
    source = "Audio Upload"
  ) => {

    const historyItem = {

      id:
        Date.now() +
        Math.random(),

      filename:
        data.filename || "Unknown Audio",

      source,

      synthetic_probability:
        data.synthetic_probability,

      genuine_probability:
        data.genuine_probability,

      maximum_synthetic_probability:
        data.maximum_synthetic_probability,

      risk_score:
        data.risk_score,

      risk_level:
        data.risk_level,

      decision:
        data.decision,

      verdict:
        data.verdict,

      suspicious_windows:
        data.suspicious_windows,

      total_windows:
        data.total_windows,

      timestamp:
        new Date().toISOString()

    }


    setHistory(
      previousHistory => [

        historyItem,

        ...previousHistory

      ].slice(0, 10)
    )

  }


  // =========================================
  // CLEAR HISTORY
  // =========================================

  const clearHistory = () => {

    setHistory([])

    localStorage.removeItem(
      "nexashield_analysis_history"
    )

  }


  // =========================================
  // AUDIO ANALYSIS
  // =========================================

  const analyzeAudio = async (
    file,
    source = "Audio Upload"
  ) => {

    if (!file) return


    setAudioFile(file)

    setResult(null)

    setError("")

    setVerificationStatus(null)

    setIsAnalyzing(true)


    const formData =
      new FormData()


    formData.append(
      "file",
      file
    )


    try {

      const response =
        await fetch(
          `${API_BASE_URL}/analyze`,
          {
            method: "POST",
            body: formData,
          }
        )


      if (!response.ok) {

        const errorData =
          await response
            .json()
            .catch(() => null)


        throw new Error(
          errorData?.detail ||
          "Analysis request failed"
        )

      }


      const data =
        await response.json()


      setResult(data)


      // =========================================
      // NO VOICE DETECTED
      // =========================================

      if (
        data.status ===
        "NO_VOICE_DETECTED"
      ) {

        // Do NOT add no-voice results
        // to analysis history.

        return

      }


      // =========================================
      // SAVE SUCCESSFUL AI ANALYSIS
      // =========================================

      addToHistory(
        data,
        source
      )


    } catch (error) {

      console.error(
        "Analysis failed:",
        error
      )


      setError(
        error.message ||
        "Unable to analyze the audio. Make sure the NexaShield backend is running."
      )

    } finally {

      await new Promise(
        resolve => setTimeout(resolve, 1500)
      )

      setIsAnalyzing(false)

    }

  }


  // =========================================
  // FILE UPLOAD
  // =========================================

  const handleFileChange = (
    event
  ) => {

    const file =
      event.target.files[0]


    if (!file) return


    setDemoMode(false)


    setAudioFile(file)


    setAudioUrl(
      URL.createObjectURL(file)
    )


    analyzeAudio(
      file,
      "Audio Upload"
    )

  }


  // =========================================
  // DEMO MODE
  // =========================================

  const runDemo = async (
    filename
  ) => {

    setDemoMode(true)

    setError("")

    setResult(null)

    setVerificationStatus(null)

    setIsAnalyzing(true)


    try {

      const response =
        await fetch(
          `${API_BASE_URL}/demo/${encodeURIComponent(filename)}`
        )


      if (!response.ok) {

        const errorData =
          await response
            .json()
            .catch(() => null)


        throw new Error(
          errorData?.detail ||
          "Demo audio could not be loaded."
        )

      }


      const audioBlob =
        await response.blob()


      const demoFile =
        new File(
          [audioBlob],
          filename,
          {
            type: "audio/wav"
          }
        )


      setAudioFile(
        demoFile
      )


      setAudioUrl(
        URL.createObjectURL(
          demoFile
        )
      )


      const source =
        filename ===
          "real_test_01.wav"

          ? "Synthetic Demo"

          : "Genuine Demo"


      await analyzeAudio(
        demoFile,
        source
      )


    } catch (error) {

      console.error(
        "Demo failed:",
        error
      )


      setIsAnalyzing(false)


      setError(
        "Unable to load the demo sample. Check that the backend is running and the demo files exist."
      )

    }

  }


  // =========================================
  // MICROPHONE RECORDING
  // =========================================

  const startRecording = async () => {

    try {

      const stream =
        await navigator.mediaDevices
          .getUserMedia({
            audio: true,
          })


      const mediaRecorder =
        new MediaRecorder(
          stream
        )


      mediaRecorderRef.current =
        mediaRecorder


      chunksRef.current = []


      mediaRecorder.ondataavailable =
        (event) => {

          if (
            event.data.size > 0
          ) {

            chunksRef.current.push(
              event.data
            )

          }

        }


      mediaRecorder.onstop =
        () => {

          const audioBlob =
            new Blob(
              chunksRef.current,
              {
                type:
                  mediaRecorder.mimeType
              }
            )


          const recordedFile =
            new File(
              [audioBlob],
              "recorded_voice.webm",
              {
                type:
                  audioBlob.type
              }
            )


          setDemoMode(false)


          setAudioFile(
            recordedFile
          )


          setAudioUrl(
            URL.createObjectURL(
              recordedFile
            )
          )


          stream
            .getTracks()
            .forEach(
              track =>
                track.stop()
            )


          analyzeAudio(
            recordedFile,
            "Voice Recording"
          )

        }


      mediaRecorder.start()


      setIsRecording(true)

      setRecordingTime(0)

      setResult(null)

      setError("")


      timerRef.current =
        setInterval(
          () => {

            setRecordingTime(
              time =>
                time + 1
            )

          },
          1000
        )


    } catch (error) {

      console.error(
        "Microphone access failed:",
        error
      )


      setError(
        "Microphone access was denied or is unavailable."
      )

    }

  }


  const stopRecording = () => {

    if (
      !mediaRecorderRef.current
    ) {

      return

    }


    mediaRecorderRef.current.stop()


    setIsRecording(false)


    clearInterval(
      timerRef.current
    )

  }


  // =========================================
  // TIME FORMAT
  // =========================================

  const formatTime = (
    seconds
  ) => {

    const minutes =
      Math.floor(
        seconds / 60
      )


    const remainingSeconds =
      seconds % 60


    return (
      String(minutes).padStart(
        2,
        "0"
      ) +
      ":" +
      String(
        remainingSeconds
      ).padStart(
        2,
        "0"
      )
    )

  }


  // =========================================
  // VERIFICATION
  // =========================================

  const handleVerification = () => {

    setVerificationStatus(
      "VERIFIED"
    )

  }


  // =========================================
  // RISK CLASS
  // =========================================

  const getRiskClass = (
    riskLevel
  ) => {

    if (!riskLevel) {

      return ""

    }


    return riskLevel.toLowerCase()

  }


  // =========================================
  // HISTORY TIME FORMAT
  // =========================================

  const formatHistoryTime = (
    timestamp
  ) => {

    try {

      return new Date(
        timestamp
      ).toLocaleString(
        undefined,
        {
          day: "2-digit",
          month: "short",
          hour: "2-digit",
          minute: "2-digit"
        }
      )

    } catch {

      return ""

    }

  }


  // =========================================
  // UI
  // =========================================

  return (

    <div className="app">


      {/* =====================================
          TOP BAR
      ===================================== */}

      <header className="topbar">

        <div>

          <h1>
            NexaShield AI
          </h1>

          <p>
            Voice Cloning Detection &
            Impersonation Prevention
          </p>

        </div>


        <div className="status">

          <span className="status-dot"></span>

          System Online

        </div>

      </header>



      <main className="dashboard">


        {/* =====================================
            DEMO MODE
        ===================================== */}

        <section className="card demo-card">

          <div className="demo-header">

            <div>

              <h2>
                🎬 Demo Mode
              </h2>

              <p className="muted">

                Try a known audio sample
                for a reproducible SIH
                demonstration.

              </p>

            </div>


            <span className="demo-badge">

              LIVE AI ANALYSIS

            </span>

          </div>


          <div className="demo-buttons">


            <button
              className="demo-btn genuine-demo"
              onClick={() =>
                runDemo(
                  "real_test_09.wav"
                )
              }
              disabled={
                isAnalyzing ||
                isRecording
              }
            >

              <span className="demo-icon">
                🟢
              </span>


              <span>

                <strong>
                  Genuine Voice
                </strong>

                <small>
                  Known genuine sample
                </small>

              </span>

            </button>



            <button
              className="demo-btn synthetic-demo"
              onClick={() =>
                runDemo(
                  "real_test_01.wav"
                )
              }
              disabled={
                isAnalyzing ||
                isRecording
              }
            >

              <span className="demo-icon">
                🔴
              </span>


              <span>

                <strong>
                  Synthetic Voice
                </strong>

                <small>
                  Known synthetic sample
                </small>

              </span>

            </button>


          </div>


          <p className="demo-note">

            ℹ️ Demo samples are processed by
            the actual NexaShield AI detection
            pipeline. Results are not hardcoded.

          </p>

        </section>



        {/* =====================================
            VOICE INPUT
        ===================================== */}

        <section className="card input-card">

          <h2>
            Voice Analysis
          </h2>


          <p className="muted">

            Upload an audio file or record a
            voice sample directly.

          </p>



          <div className="upload-area">


            <div className="upload-icon">
              🎙️
            </div>


            <h3>

              {isRecording
                ? "Recording Voice..."
                : "Voice Sample"}

            </h3>


            {isRecording ? (

              <>

                <div className="recording-indicator">

                  <span className="recording-dot"></span>

                  Recording{" "}
                  {formatTime(
                    recordingTime
                  )}

                </div>


                <button
                  className="stop-btn"
                  onClick={
                    stopRecording
                  }
                >

                  ⏹ Stop Recording

                </button>

              </>

            ) : (

              <>

                <p>

                  {audioFile
                    ? `Selected: ${audioFile.name}`
                    : "Upload a file or record directly"}

                </p>


                <div className="button-row">


                  <label className="primary-btn">

                    📁 Choose Audio


                    <input
                      type="file"
                      accept="audio/*"
                      onChange={
                        handleFileChange
                      }
                      hidden
                    />

                  </label>



                  <button
                    className="record-btn"
                    onClick={
                      startRecording
                    }
                    disabled={
                      isAnalyzing
                    }
                  >

                    🔴 Record Voice

                  </button>


                </div>

              </>

            )}

          </div>



          {isAnalyzing && (

            <div className="analysis-pipeline">

              <div className="pipeline-title">
                🔄 NexaShield AI is analyzing...
              </div>

              <div className="pipeline">

                <div className="pipeline-step active">
                  <span className="pipeline-icon">🎙️</span>
                  <span>Audio Received</span>
                </div>

                <div className="pipeline-line"></div>

                <div className="pipeline-step active">
                  <span className="pipeline-icon">🔊</span>
                  <span>Voice Activity Check</span>
                </div>

                <div className="pipeline-line"></div>

                <div className="pipeline-step active">
                  <span className="pipeline-icon">🧠</span>
                  <span>AI Voice Analysis</span>
                </div>

                <div className="pipeline-line"></div>

                <div className="pipeline-step active">
                  <span className="pipeline-icon">🛡️</span>
                  <span>Risk Assessment</span>
                </div>

              </div>

            </div>

          )}



          {error && (

            <div className="error-message">

              ⚠️ {error}

            </div>

          )}

        </section>



        {/* =====================================
            AUDIO PLAYER
        ===================================== */}

        {audioUrl && (

          <section className="card audio-player-card">

            <h2>
              🔊 Audio Sample
            </h2>


            <p className="muted">

              {demoMode
                ? "Demo sample being analyzed by NexaShield AI."
                : "Listen to the audio analyzed by NexaShield AI."}

            </p>


            <audio
              controls
              src={audioUrl}
              style={{
                width: "100%"
              }}
            >

              Your browser does not
              support the audio player.

            </audio>

          </section>

        )}



        {/* =====================================
            DETECTION RESULT
        ===================================== */}

        <section className="card">

          <h2>
            Detection Result
          </h2>


          {result?.status ===
            "NO_VOICE_DETECTED" ? (

            <div className="no-voice-result">

              <div className="no-voice-icon">
                🔇
              </div>

              <h3>
                No Voice Detected
              </h3>

              <p>
                {result.message ||
                  "NexaShield could not identify sufficient speech activity in this recording. Please record or upload a voice sample containing speech."}
              </p>

              <div className="no-voice-status">
                ANALYSIS NOT PERFORMED
              </div>

            </div>

          ) : result ? (

            <div className="result-content">


              <div className="result-number">

                {result.synthetic_probability}%

              </div>


              <div className="result-label">

                Synthetic Probability

              </div>


              <div className="probability-row">

                <span>
                  Genuine
                </span>

                <strong>
                  {result.genuine_probability}%
                </strong>

              </div>


              <div className="probability-row">

                <span>
                  Synthetic
                </span>

                <strong>
                  {result.synthetic_probability}%
                </strong>

              </div>


              <div className="peak-result">

                <span>
                  Peak Synthetic Probability
                </span>

                <strong>
                  {result.maximum_synthetic_probability}%
                </strong>

              </div>


            </div>

          ) : (

            <div className="result-placeholder">

              <div className="result-number">
                --%
              </div>


              <p>

                {isAnalyzing
                  ? "Analyzing voice..."
                  : "Upload, record, or try a demo sample to begin analysis"}

              </p>

            </div>

          )}

        </section>



        {/* =====================================
            RISK ASSESSMENT
        ===================================== */}

        <section className="card">

          <h2>
            Risk Assessment
          </h2>


          {result?.status ===
            "NO_VOICE_DETECTED" ? (

            <div className="risk-placeholder">

              <div className="risk-level">
                NOT ASSESSED
              </div>

              <p>
                No security risk was calculated because
                sufficient voice activity was not detected.
              </p>

            </div>

          ) : result ? (

            <div className="risk-content">


              <div
                className={
                  "risk-level " +
                  getRiskClass(
                    result.risk_level
                  )
                }
              >

                {result.risk_level}

              </div>

              <div className="risk-meter">
                <div className="risk-meter-header">
                  <span>Threat Level</span>
                  <strong>{result.risk_score}/100</strong>
                </div>

                <div className="risk-meter-track">
                  <div
                    className={
                      "risk-meter-fill " +
                      getRiskClass(result.risk_level)
                    }
                    style={{
                      width: `${result.risk_score}%`
                    }}
                  ></div>
                </div>

                <div className="risk-meter-labels">
                  <span>LOW</span>
                  <span>MEDIUM</span>
                  <span>HIGH</span>
                  <span>CRITICAL</span>
                </div>
              </div>


              <div className="risk-score">

                <span>
                  Risk Score
                </span>

                <strong>
                  {result.risk_score}
                </strong>

              </div>


              <div className="decision">

                <span>
                  Security Decision
                </span>

                <strong>
                  {result.decision}
                </strong>

              </div>


              <p className="recommendation">

                {result.recommendation}

              </p>

            </div>

          ) : (

            <div className="risk-placeholder">

              <div className="risk-level">
                WAITING
              </div>

              <p>
                No voice sample analyzed yet.
              </p>

            </div>

          )}

        </section>



        {/* =====================================
            SECURITY ACTION
        ===================================== */}

        {result &&
          result.status !==
          "NO_VOICE_DETECTED" && (

            <section className="card prevention-card">

              <h2>
                Security Action
              </h2>


              {result.decision === "ALLOW" && (

                <div className="action-content allow-action">

                  <div className="action-icon">
                    🟢
                  </div>


                  <h3>
                    Voice Assessment Passed
                  </h3>


                  <p>

                    No significant synthetic
                    characteristics detected by
                    the AI analysis.

                  </p>


                  <div className="action-status">

                    ✓ ALLOWED

                  </div>

                </div>

              )}



              {result.decision === "VERIFY" && (

                <div className="action-content verify-action">

                  <div className="action-icon">
                    🟡
                  </div>


                  <h3>
                    Additional Verification Required
                  </h3>


                  <p>

                    Voice authenticity could not be
                    established with sufficient confidence.

                  </p>


                  {verificationStatus === "VERIFIED" ? (

                    <div className="action-status verified">

                      ✓ IDENTITY VERIFIED

                    </div>

                  ) : (

                    <button
                      className="verify-btn"
                      onClick={
                        handleVerification
                      }
                    >

                      🔐 Verify Identity

                    </button>

                  )}

                </div>

              )}



              {result.decision === "BLOCK" && (

                <div className="action-content block-action">

                  <div className="action-icon">
                    🔴
                  </div>


                  <h3>
                    Potential Voice Impersonation Detected
                  </h3>


                  <p>

                    This interaction should be
                    blocked or placed on hold.

                  </p>


                  <div className="action-status blocked">

                    🚫 TRANSACTION BLOCKED

                  </div>

                  <div className="prevention-flow">
                    <div className="prevention-flow-title">
                      🛡️ NexaShield Prevention
                    </div>

                    <div className="prevention-steps">
                      <div className="prevention-step completed">
                        <span>✓</span>
                        <span>Threat Detected</span>
                      </div>

                      <div className="prevention-step completed">
                        <span>✓</span>
                        <span>Risk Evaluated</span>
                      </div>

                      <div className="prevention-step blocked">
                        <span>🔒</span>
                        <span>Interaction Blocked</span>
                      </div>
                    </div>

                    <p className="prevention-note">
                      NexaShield prevented the suspicious interaction
                      from proceeding without additional verification.
                    </p>
                  </div>

                </div>

              )}


            </section>

          )}



        {/* =====================================
            WINDOW ANALYSIS
        ===================================== */}

        {result &&
          result.status !==
          "NO_VOICE_DETECTED" &&
          result.windows && (

            <section className="card window-card">

              <div className="window-header">

                <div>

                  <h2>
                    Window Analysis
                  </h2>


                  <p className="muted">

                    {result.suspicious_windows} of{" "}
                    {result.total_windows} windows
                    flagged as suspicious

                  </p>

                </div>

              </div>


              <div className="window-list">


                {result.windows.map(
                  (window) => {

                    const synthetic =
                      window.synthetic_probability *
                      100


                    const suspicious =
                      synthetic >= 70


                    return (

                      <div
                        className="window-row"
                        key={
                          window.window
                        }
                      >


                        <div className="window-name">

                          Window{" "}
                          {window.window}

                        </div>


                        <div className="window-bar-container">

                          <div
                            className={
                              suspicious
                                ? "window-bar suspicious"
                                : "window-bar"
                            }
                            style={{
                              width:
                                `${synthetic}%`
                            }}
                          ></div>

                        </div>


                        <div className="window-value">

                          {synthetic.toFixed(
                            2
                          )}%

                        </div>


                      </div>

                    )

                  }
                )}


              </div>

            </section>

          )}



        {/* =====================================
            ANALYSIS HISTORY
        ===================================== */}

        <section className="card history-card">

          <div className="history-header">

            <div>

              <h2>
                🗂️ Analysis History
              </h2>

              <p className="muted">

                Last {history.length} analysis
                {history.length === 1
                  ? ""
                  : "es"} stored locally.

              </p>

            </div>


            {history.length > 0 && (

              <button
                className="clear-history-btn"
                onClick={
                  clearHistory
                }
              >

                Clear History

              </button>

            )}

          </div>



          {history.length === 0 ? (

            <div className="history-empty">

              <div className="history-empty-icon">
                📊
              </div>

              <strong>
                No analyses yet
              </strong>

              <p>
                Your recent voice assessments
                will appear here.
              </p>

            </div>

          ) : (

            <div className="history-list">

              {history.map(
                (item) => {

                  const riskClass =
                    getRiskClass(
                      item.risk_level
                    )


                  const decisionIcon =
                    item.decision === "ALLOW"
                      ? "🟢"
                      : item.decision === "VERIFY"
                        ? "🟡"
                        : "🔴"


                  return (

                    <div
                      className="history-item"
                      key={item.id}
                    >


                      <div className="history-main">

                        <div className="history-icon">

                          {decisionIcon}

                        </div>


                        <div className="history-details">

                          <strong>
                            {item.filename}
                          </strong>


                          <span>

                            {item.source}

                          </span>


                          <small>

                            {formatHistoryTime(
                              item.timestamp
                            )}

                          </small>

                        </div>

                      </div>



                      <div className="history-metrics">

                        <div>

                          <span>
                            Synthetic
                          </span>

                          <strong>
                            {item.synthetic_probability}%
                          </strong>

                        </div>


                        <div>

                          <span>
                            Risk
                          </span>

                          <strong>
                            {item.risk_score}
                          </strong>

                        </div>


                        <div>

                          <span>
                            Level
                          </span>

                          <strong
                            className={
                              `history-risk ${riskClass}`
                            }
                          >
                            {item.risk_level}
                          </strong>

                        </div>


                        <div>

                          <span>
                            Decision
                          </span>

                          <strong>
                            {item.decision}
                          </strong>

                        </div>

                      </div>


                    </div>

                  )

                }
              )}

            </div>

          )}

        </section>


      </main>

    </div>

  )

}


export default App