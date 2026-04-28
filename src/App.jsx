// npm run dev -- --host

import { useState, useRef } from 'react';
import axios from 'axios';
import exifr from 'exifr';

const FONTS = `@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');`;

const GLOBAL_CSS = `
  * { box-sizing: border-box; }
  body { margin: 0; background: #080b0f; }
  @keyframes spin    { to { transform: rotate(360deg); } }
  @keyframes fadeUp  { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  @keyframes blink   { 0%,100%{opacity:1;} 50%{opacity:0;} }
  @keyframes scanline { 0%{transform:translateY(-100%);} 100%{transform:translateY(100vh);} }
  .scan-overlay {
    pointer-events:none; position:fixed; inset:0; z-index:0;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,
      rgba(0,255,160,.012) 2px,rgba(0,255,160,.012) 4px);
  }
  .fadeUp { animation: fadeUp .45s cubic-bezier(.22,1,.36,1) both; }
  .amber-glow { text-shadow: 0 0 18px rgba(255,180,0,.55); }
  ::-webkit-scrollbar{width:4px;}
  ::-webkit-scrollbar-track{background:#0d1117;}
  ::-webkit-scrollbar-thumb{background:#2a3040;border-radius:4px;}
  .hist-item:hover{background:#141b26!important;}
  .upload-btn:hover{filter:brightness(1.12);}
  .upload-btn:active{transform:scale(.97);}
  .ev-row:hover{background:#0f1520!important;}
`;

const CATS = {
  "Restaurant":        {icon:"🍽️",accent:"#ff6b35"},
  "Cafe":              {icon:"☕", accent:"#c8933a"},
  "Bar":               {icon:"🍺",accent:"#e2b040"},
  "Hotel":             {icon:"🏨",accent:"#7c6fe0"},
  "Shop / Market":     {icon:"🛍️",accent:"#34c985"},
  "Pharmacy":          {icon:"💊",accent:"#e05c5c"},
  "Bank / ATM":        {icon:"🏦",accent:"#4a9eff"},
  "Hospital / Clinic": {icon:"🏥",accent:"#ef4444"},
  "School":            {icon:"🎓",accent:"#a78bfa"},
  "Mosque / Church":   {icon:"🕌",accent:"#c084fc"},
  "Residential":       {icon:"🏠",accent:"#64748b"},
  "Gas Station":       {icon:"⛽",accent:"#fb923c"},
  "Museum / Culture":  {icon:"🏛️",accent:"#d4a017"},
  "Park / Nature":     {icon:"🌿",accent:"#22c55e"},
  "Transport":         {icon:"🚉",accent:"#38bdf8"},
  "Place":             {icon:"📍",accent:"#ffb400"},
};
const getCat = (k) => CATS[k] || CATS["Place"];

// Confidence colours for VLM per-sign confidence (1-3)
const CONF_COLORS = { 1: "#ef4444", 2: "#e2b040", 3: "#22c55e" };
const CONF_LABELS = { 1: "LOW", 2: "MED", 3: "HIGH" };

// Location-source metadata
const SRC_META = {
  gps:   {color:"#22c55e", icon:"📡", label:"EXIF GPS",    bar: 100},
  signs: {color:"#ffb400", icon:"🔎", label:"Sign Match",  bar: null},  // bar from score
};

// ── Client-side resize ────────────────────────────────────────────────────────
const resizeImage = (file, maxSide = 1600) => new Promise((resolve) => {
  const img = new Image();
  const url = URL.createObjectURL(file);
  img.onload = () => {
    URL.revokeObjectURL(url);
    const {naturalWidth:w, naturalHeight:h} = img;
    const scale = Math.min(1, maxSide / Math.max(w, h));
    if (scale >= 1) { resolve(file); return; }
    const canvas = document.createElement("canvas");
    canvas.width  = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => resolve(new File([blob], file.name, {type:"image/jpeg"})),
      "image/jpeg", 0.92
    );
  };
  img.onerror = () => { URL.revokeObjectURL(url); resolve(file); };
  img.src = url;
});

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [preview,   setPreview]   = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [status,    setStatus]    = useState("");
  const [result,    setResult]    = useState(null);
  const [dragOver,  setDragOver]  = useState(false);
  const [history,   setHistory]   = useState([]);
  const [showDebug, setShowDebug] = useState(false);

  const galleryRef = useRef(null);
  const cameraRef  = useRef(null);

  const processFile = async (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const thumbUrl = URL.createObjectURL(file);
    setPreview(thumbUrl);
    setResult(null);
    setLoading(true);
    let lat = null, lng = null;

    // 1. Read GPS from original before resize strips EXIF
    try {
      setStatus("Reading EXIF GPS…");
      const exif = await exifr.parse(file, {gps:true});
      if (exif?.latitude) {
        lat = exif.latitude;
        lng = exif.longitude;
        setStatus("GPS found — using as primary location…");
      } else {
        setStatus("No GPS — will use sign intersection…");
      }
    } catch (_) {}

    // 2. Resize on-device
    let uploadFile = file;
    try {
      uploadFile = await resizeImage(file, 1600);
    } catch (_) {}

    // 3. Send to backend
    try {
      setStatus(lat ? "GPS confirmed · VLM scanning scene…" : "VLM scanning signs…");
      const fd = new FormData();
      fd.append("image", uploadFile);
      if (lat !== null) { fd.append("lat", lat); fd.append("lng", lng); }

      if (!lat) setStatus("Running intersection engine…");

      const {data} = await axios.post(
        `${import.meta.env.VITE_API_URL}/analyze-scene/`,
        fd, {headers:{"Content-Type":"multipart/form-data"}}
      );
      const full = {...data, clientGpsUsed: lat !== null};
      setResult(full);
      setHistory(h => [{thumb:thumbUrl, result:full, ts:Date.now()}, ...h].slice(0,8));
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || "Unknown error";
      setResult({status:"error", message:`Backend: ${msg}`});
    } finally {
      setLoading(false);
      setStatus("");
    }
  };

  const onFileChange = (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) processFile(file);
  };

  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    processFile(e.dataTransfer.files[0]);
  };

  const cat      = result?.status === "success" ? getCat(result.place_data?.category) : null;
  const hasMap   = result?.status === "success" && result.place_data?.lat !== 0;
  const locSrc   = result?.location_source;
  const srcMeta  = locSrc ? SRC_META[locSrc] : null;

  // Confidence bar pct: GPS=100, signs=score capped to 0-100
  const confPct = locSrc === "gps" ? 100
    : Math.min(100, Math.round((result?.confidence_score || 0) / 6 * 100));

  const confColor = result?.confidence_label === "High"    ? "#22c55e"
                  : result?.confidence_label === "Medium"  ? "#e2b040"
                  : result?.confidence_label === "GPS Exact"? "#22c55e"
                  : "#f97316";

  return (
    <>
      <style>{FONTS}</style>
      <style>{GLOBAL_CSS}</style>
      <div className="scan-overlay" />
      <div style={s.root}>

        {/* ── Header ── */}
        <header style={s.header}>
          <div style={s.headerInner}>
            <div style={s.logoMark}>
              <span style={s.logoIcon}>◈</span>
              <div>
                <div style={s.logoTitle}>LOCATION LENS</div>
                <div style={s.logoSub}>VLM · INTERSECTION ENGINE · OSM</div>
              </div>
            </div>
            <div style={s.headerDot} />
          </div>
        </header>

        <main style={s.main}>

          {/* hidden inputs */}
          <input ref={galleryRef} type="file" accept="image/*"
            onChange={onFileChange} style={{display:"none"}} />
          <input ref={cameraRef} type="file" accept="image/*" capture="environment"
            onChange={onFileChange} style={{display:"none"}} />

          {/* drop zone */}
          <div
            style={{...s.dropzone,...(dragOver?s.dropActive:{}),...(preview?s.dropSmall:{})}}
            onDragOver={(e)=>{e.preventDefault();setDragOver(true);}}
            onDragLeave={()=>setDragOver(false)}
            onDrop={onDrop}
          >
            {preview ? (
              <>{<img src={preview} alt="preview" style={s.previewImg}/>}
                {loading && <div style={s.scanBar}/>}
              </>
            ) : (
              <div style={s.dropContent}>
                <div style={s.dropBracket}><span style={s.dropIconText}>[ FRAME TARGET ]</span></div>
                <p style={s.dropHint}>drag & drop · gallery · camera</p>
              </div>
            )}
          </div>

          {/* buttons */}
          <div style={s.btnRow}>
            <button className="upload-btn" style={s.btnGallery}
              onClick={()=>galleryRef.current?.click()} disabled={loading}>
              <span style={s.btnIcon}>◧</span> GALLERY
            </button>
            <button className="upload-btn" style={s.btnCamera}
              onClick={()=>cameraRef.current?.click()} disabled={loading}>
              <span style={s.btnIcon}>⊙</span> CAMERA
            </button>
          </div>

          {/* loading */}
          {loading && (
            <div style={s.loadCard}>
              <div style={s.spinner}/>
              <div>
                <div style={s.loadStatus}>
                  {status}<span style={{animation:"blink 1s step-end infinite"}}>█</span>
                </div>
                <div style={s.loadHint}>Qwen2.5-VL + intersection engine</div>
              </div>
            </div>
          )}

          {/* ── SUCCESS ── */}
          {result?.status === "success" && (
            <div className="fadeUp" style={s.resultCard}>

              {/* Scene */}
              {result.scene && (
                <div style={s.sceneBox}>
                  <span style={s.sceneLabel}>SCENE</span>
                  <span style={s.sceneText}>{result.scene}</span>
                </div>
              )}

              {/* ── LOCATION SOURCE + CONFIDENCE ── */}
              <div style={s.confBlock}>
                <div style={s.confTopRow}>

                  {/* Source badge */}
                  <div style={{...s.srcBadge,
                    background: srcMeta?.color + "18",
                    borderColor: srcMeta?.color + "55",
                    color: srcMeta?.color}}>
                    <span>{srcMeta?.icon}</span>
                    <span style={s.srcLabel}>{srcMeta?.label}</span>
                  </div>

                  {/* Confidence label */}
                  <span style={{...s.confLabel, color: confColor}}>
                    {result.confidence_label}
                  </span>
                </div>

                {/* Bar */}
                <div style={s.barTrack}>
                  <div style={{
                    ...s.barFill,
                    width: `${confPct}%`,
                    background: confColor,
                    boxShadow: `0 0 6px ${confColor}88`,
                    transition: "width .9s cubic-bezier(.22,1,.36,1)",
                  }}/>
                </div>

                {/* GPS path: clean explanation */}
                {locSrc === "gps" && (
                  <p style={s.gpsNote}>
                    Location from EXIF GPS coordinates embedded in this photo.
                    Sign names are used for categorisation only.
                  </p>
                )}

                {/* Intersection path: evidence table */}
                {locSrc === "signs" && result.evidence?.length > 0 && (
                  <div style={{marginTop:10}}>
                    <div style={s.evHeader}>
                      {result.evidence.length === 1
                        ? "1 sign matched — no corroboration"
                        : `${result.evidence.length} signs placed this location:`}
                    </div>
                    <div style={s.evList}>
                      {result.evidence.map((ev, i) => (
                        <div key={i} className="ev-row" style={s.evRow}>
                          <span style={{...s.evRole,
                            color: ev.role === "anchor" ? "#ffb400" : "#22c55e"}}>
                            {ev.role === "anchor" ? "⚓" : "✓"}
                          </span>
                          <span style={s.evName}>{ev.name}</span>
                          {ev.confidence && (
                            <span style={{...s.evConf,
                              color: CONF_COLORS[ev.confidence],
                              borderColor: CONF_COLORS[ev.confidence] + "44"}}>
                              {CONF_LABELS[ev.confidence]}
                            </span>
                          )}
                          {ev.dist_m !== undefined && (
                            <span style={s.evDist}>{ev.dist_m}m away</span>
                          )}
                        </div>
                      ))}
                    </div>
                    {result.evidence.length > 1 && (
                      <p style={s.evNote}>
                        These signs were all found within 200 m of each other in OpenStreetMap,
                        uniquely identifying this location.
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* category badge */}
              <div style={{...s.catBadge, background:cat.accent+"20", borderColor:cat.accent+"55"}}>
                <span>{cat.icon}</span>
                <span style={{color:cat.accent, fontFamily:"'Space Mono',monospace",
                              fontSize:11, letterSpacing:"0.1em"}}>
                  {result.place_data.category.toUpperCase()}
                </span>
              </div>

              <h2 style={{...s.placeName, color:cat.accent}} className="amber-glow">
                {result.place_data.name}
              </h2>
              <p style={s.placeAddr}>{result.place_data.formatted_address}</p>

              {/* chips */}
              <div style={s.chipRow}>
                {result.place_data.city    && <Chip>🏙 {result.place_data.city}</Chip>}
                {result.place_data.country && <Chip>🌍 {result.place_data.country}</Chip>}
                {result.place_data.type    && <Chip>🏷 {result.place_data.type}</Chip>}
              </div>

              {/* map links */}
              {hasMap && (
                <div style={s.mapRow}>
                  <MapLink href={`https://www.openstreetmap.org/?mlat=${result.place_data.lat}&mlon=${result.place_data.lng}&zoom=18`}>OSM ↗</MapLink>
                  <MapLink href={`https://www.google.com/maps?q=${result.place_data.lat},${result.place_data.lng}`} color="#4285f4">Google Maps ↗</MapLink>
                  <MapLink href={`https://maps.apple.com/?ll=${result.place_data.lat},${result.place_data.lng}&q=${encodeURIComponent(result.place_data.name)}`} color="#666">Apple Maps ↗</MapLink>
                </div>
              )}

              <div style={s.rule}/>

              {/* debug toggle */}
              <button style={s.debugToggle} onClick={()=>setShowDebug(v=>!v)}>
                {showDebug?"▲":"▼"} VLM DEBUG · method: {result.search_mode}
              </button>

              {showDebug && (
                <div style={s.debugPanel}>
                  <div style={s.debugRow}>
                    <span style={s.debugKey}>RAW READ</span>
                    <span style={s.debugVal}>"{result.extracted_text}"</span>
                  </div>
                  <div style={s.debugRow}>
                    <span style={s.debugKey}>CONF SCORE</span>
                    <span style={s.debugVal}>{result.confidence_score}</span>
                  </div>
                  {result.ocr_lines?.length > 0 && (
                    <>
                      <div style={{...s.debugKey, marginTop:10, marginBottom:4}}>ALL SIGNS DETECTED BY VLM</div>
                      <table style={s.table}>
                        <thead>
                          <tr>
                            <th style={s.th}>#</th>
                            <th style={s.th}>SIGN NAME</th>
                            <th style={s.th}>VLM CONF</th>
                            <th style={s.th}>VOTED</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.ocr_lines.map((l,i) => {
                            const voted = result.evidence?.some(e => e.name === l.text);
                            const cColor = CONF_COLORS[l.confidence] || "#64748b";
                            return (
                              <tr key={i}>
                                <td style={s.td}>{i+1}</td>
                                <td style={{...s.td, fontWeight:600,
                                  color: voted?"#ffb400":"#e2e8f0"}}>{l.text}</td>
                                <td style={{...s.td, color:cColor, fontWeight:700}}>
                                  {l.conf_label || l.confidence}
                                </td>
                                <td style={{...s.td, color:voted?"#22c55e":"#334155"}}>
                                  {voted?"✓":"—"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── ERROR ── */}
          {result?.status === "error" && (
            <div className="fadeUp" style={s.errorCard}>
              <div style={s.errorHeader}>
                <span style={s.errorX}>✕</span>
                <span style={s.errorTitle}>NO MATCH</span>
              </div>
              <p style={s.errorMsg}>{result.message}</p>
              {result.scene && (
                <div style={{...s.sceneBox, marginTop:12}}>
                  <span style={s.sceneLabel}>SCENE</span>
                  <span style={s.sceneText}>{result.scene}</span>
                </div>
              )}
              {result.ocr_lines?.length > 0 && (
                <>
                  <div style={s.rule}/>
                  <div style={{...s.debugKey, marginBottom:6}}>SIGNS DETECTED (NOT IN OSM)</div>
                  <table style={s.table}>
                    <thead>
                      <tr>
                        <th style={s.th}>#</th>
                        <th style={s.th}>SIGN NAME</th>
                        <th style={s.th}>VLM CONF</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.ocr_lines.map((l,i) => {
                        const cColor = CONF_COLORS[l.confidence] || "#64748b";
                        return (
                          <tr key={i}>
                            <td style={s.td}>{i+1}</td>
                            <td style={{...s.td, color:"#e2e8f0", fontWeight:600}}>{l.text}</td>
                            <td style={{...s.td, color:cColor, fontWeight:700}}>
                              {l.conf_label || "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p style={s.tip}>
                    💡 Signs were read but no corroborating location was found in OpenStreetMap.
                    The place may not be indexed yet. Try enabling GPS before taking the photo.
                  </p>
                </>
              )}
            </div>
          )}

          {/* ── HISTORY ── */}
          {history.length > 0 && (
            <div style={s.histSection}>
              <div style={s.histHeader}>
                <span style={s.histLabel}>SCAN HISTORY</span>
                <button style={s.histClear} onClick={()=>setHistory([])}>CLEAR</button>
              </div>
              <div style={s.histList}>
                {history.map((h,i) => {
                  const hCat = h.result?.status==="success"
                    ? getCat(h.result.place_data?.category) : null;
                  const hSrc = h.result?.location_source;
                  return (
                    <div key={i} className="hist-item" style={s.histItem}
                      onClick={()=>{setPreview(h.thumb);setResult(h.result);}}>
                      <img src={h.thumb} alt="" style={s.histThumb}/>
                      <div style={s.histInfo}>
                        {h.result?.status==="success" ? (
                          <>
                            <div style={{...s.histName, color:hCat.accent}}>
                              {h.result.place_data.name}
                            </div>
                            <div style={s.histMeta}>
                              <span style={s.histCat}>{hCat.icon} {h.result.place_data.category}</span>
                              <span style={{...s.histSrc,
                                color: hSrc==="gps"?"#22c55e":"#e2b040"}}>
                                {hSrc==="gps"?"📡":"🔎"} {h.result.confidence_label}
                              </span>
                            </div>
                          </>
                        ) : (
                          <div style={s.histName}>No match</div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

        </main>
      </div>
    </>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function Chip({children, color}) {
  return (
    <span style={{background:"#0d1117", border:"1px solid #1e2a3a", borderRadius:4,
      padding:"3px 9px", fontSize:11, color:color||"#64748b",
      fontFamily:"'Space Mono',monospace"}}>
      {children}
    </span>
  );
}

function MapLink({href, children, color="#ffb400"}) {
  return (
    <a href={href} target="_blank" rel="noreferrer" style={{
      padding:"7px 14px", borderRadius:6, border:`1px solid ${color}44`,
      color, fontSize:12, fontWeight:700, fontFamily:"'Space Mono',monospace",
      textDecoration:"none", background:color+"12",
    }}>
      {children}
    </a>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const s = {
  root:{minHeight:"100vh",background:"#080b0f",fontFamily:"'DM Sans',sans-serif",
    color:"#c8d4e0",position:"relative",zIndex:1},
  header:{borderBottom:"1px solid #131c28",padding:"0 20px",
    background:"rgba(8,11,15,.92)",backdropFilter:"blur(12px)",
    position:"sticky",top:0,zIndex:100},
  headerInner:{maxWidth:680,margin:"0 auto",height:56,
    display:"flex",alignItems:"center",justifyContent:"space-between"},
  logoMark:{display:"flex",alignItems:"center",gap:12},
  logoIcon:{fontSize:22,color:"#ffb400",textShadow:"0 0 14px rgba(255,180,0,.7)",fontFamily:"monospace"},
  logoTitle:{fontFamily:"'Space Mono',monospace",fontSize:13,fontWeight:700,letterSpacing:"0.14em",color:"#e2e8f0"},
  logoSub:{fontFamily:"'Space Mono',monospace",fontSize:9,color:"#334155",letterSpacing:"0.1em",marginTop:1},
  headerDot:{width:8,height:8,borderRadius:"50%",background:"#22c55e",boxShadow:"0 0 8px #22c55e"},

  main:{maxWidth:680,margin:"0 auto",padding:"24px 16px 80px",display:"flex",flexDirection:"column",gap:14},

  dropzone:{border:"1px solid #1a2332",borderRadius:10,background:"#0d1117",
    display:"flex",alignItems:"center",justifyContent:"center",
    minHeight:200,overflow:"hidden",cursor:"pointer",position:"relative",transition:"border-color .2s"},
  dropActive:{borderColor:"#ffb400",boxShadow:"0 0 0 1px #ffb40044"},
  dropSmall:{minHeight:0},
  dropContent:{textAlign:"center",padding:"40px 20px"},
  dropBracket:{border:"1px solid #1e2a3a",borderRadius:6,padding:"12px 24px",display:"inline-block",marginBottom:12},
  dropIconText:{fontFamily:"'Space Mono',monospace",fontSize:13,color:"#334155",letterSpacing:"0.12em"},
  dropHint:{margin:0,fontSize:12,color:"#2a3a4a",fontFamily:"'Space Mono',monospace",letterSpacing:"0.08em"},
  previewImg:{width:"100%",maxHeight:380,objectFit:"cover",display:"block"},
  scanBar:{position:"absolute",left:0,right:0,height:2,
    background:"linear-gradient(90deg,transparent,#ffb400,transparent)",
    boxShadow:"0 0 12px #ffb400",animation:"scanline 1.8s linear infinite"},

  btnRow:{display:"flex",gap:10},
  btnGallery:{flex:1,padding:"12px 0",borderRadius:8,border:"1px solid #1e2a3a",
    background:"#0d1117",color:"#94a3b8",fontFamily:"'Space Mono',monospace",
    fontSize:12,fontWeight:700,letterSpacing:"0.1em",cursor:"pointer",
    display:"flex",alignItems:"center",justifyContent:"center",gap:8,transition:"filter .15s,transform .1s"},
  btnCamera:{flex:1,padding:"12px 0",borderRadius:8,border:"1px solid #ffb40033",
    background:"#ffb40010",color:"#ffb400",fontFamily:"'Space Mono',monospace",
    fontSize:12,fontWeight:700,letterSpacing:"0.1em",cursor:"pointer",
    display:"flex",alignItems:"center",justifyContent:"center",gap:8,transition:"filter .15s,transform .1s"},
  btnIcon:{fontSize:15},

  loadCard:{background:"#0d1117",borderRadius:10,border:"1px solid #131c28",
    padding:"20px 24px",display:"flex",alignItems:"center",gap:16},
  spinner:{width:24,height:24,flexShrink:0,border:"2px solid #1e2a3a",
    borderTop:"2px solid #ffb400",borderRadius:"50%",animation:"spin .7s linear infinite"},
  loadStatus:{fontFamily:"'Space Mono',monospace",fontSize:13,color:"#ffb400"},
  loadHint:{fontSize:12,color:"#334155",marginTop:3},

  resultCard:{background:"#0a0e14",borderRadius:12,padding:"22px 22px 18px",border:"1px solid #131c28"},
  sceneBox:{background:"#0d1117",borderRadius:6,border:"1px solid #1a2332",
    padding:"10px 14px",marginBottom:16,display:"flex",gap:10,alignItems:"flex-start"},
  sceneLabel:{fontFamily:"'Space Mono',monospace",fontSize:9,color:"#ffb400",
    letterSpacing:"0.14em",marginTop:1,flexShrink:0},
  sceneText:{fontSize:13,color:"#94a3b8",lineHeight:1.55},

  // Confidence / source block
  confBlock:{background:"#0d1117",borderRadius:8,border:"1px solid #1a2332",
    padding:"14px 16px",marginBottom:16},
  confTopRow:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10},
  srcBadge:{display:"inline-flex",alignItems:"center",gap:6,padding:"3px 10px",
    borderRadius:4,border:"1px solid",fontFamily:"'Space Mono',monospace",fontSize:11,fontWeight:700},
  srcLabel:{letterSpacing:"0.08em"},
  confLabel:{fontFamily:"'Space Mono',monospace",fontSize:12,fontWeight:700,letterSpacing:"0.08em"},
  barTrack:{height:3,background:"#1a2332",borderRadius:2,overflow:"hidden",marginBottom:12},
  barFill:{height:"100%",borderRadius:2},
  gpsNote:{margin:0,fontSize:11,color:"#334155",lineHeight:1.6,fontStyle:"italic"},

  // Evidence list
  evHeader:{fontFamily:"'Space Mono',monospace",fontSize:9,color:"#334155",
    letterSpacing:"0.1em",marginBottom:7},
  evList:{display:"flex",flexDirection:"column",gap:4},
  evRow:{display:"flex",alignItems:"center",gap:8,padding:"5px 8px",
    borderRadius:4,background:"#080b0f",transition:"background .15s"},
  evRole:{fontSize:13,flexShrink:0},
  evName:{flex:1,fontSize:12,color:"#94a3b8",fontFamily:"'Space Mono',monospace",
    overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},
  evConf:{fontFamily:"'Space Mono',monospace",fontSize:9,padding:"1px 6px",
    borderRadius:3,border:"1px solid",fontWeight:700,flexShrink:0},
  evDist:{fontSize:10,color:"#334155",flexShrink:0},
  evNote:{margin:"10px 0 0",fontSize:11,color:"#334155",lineHeight:1.6,fontStyle:"italic"},

  catBadge:{display:"inline-flex",alignItems:"center",gap:7,border:"1px solid",
    borderRadius:4,padding:"3px 10px",marginBottom:10},
  placeName:{margin:"0 0 6px",fontSize:26,fontWeight:700,
    fontFamily:"'Space Mono',monospace",lineHeight:1.2,letterSpacing:"-0.01em"},
  placeAddr:{margin:"0 0 14px",fontSize:12,color:"#334155",lineHeight:1.6},
  chipRow:{display:"flex",flexWrap:"wrap",gap:7,marginBottom:16},
  mapRow:{display:"flex",flexWrap:"wrap",gap:8,marginBottom:16},
  rule:{border:"none",borderTop:"1px solid #131c28",margin:"14px 0"},

  debugToggle:{background:"none",border:"1px solid #1e2a3a",borderRadius:4,
    padding:"4px 12px",color:"#334155",fontSize:10,cursor:"pointer",
    fontFamily:"'Space Mono',monospace",letterSpacing:"0.1em"},
  debugPanel:{marginTop:12,background:"#0d1117",borderRadius:6,
    border:"1px solid #131c28",padding:"12px 14px"},
  debugRow:{display:"flex",gap:12,marginBottom:6},
  debugKey:{fontFamily:"'Space Mono',monospace",fontSize:9,color:"#334155",
    letterSpacing:"0.12em",flexShrink:0,textTransform:"uppercase",marginTop:1},
  debugVal:{fontSize:12,color:"#64748b",wordBreak:"break-word"},
  table:{width:"100%",borderCollapse:"collapse",fontSize:11,marginTop:10,
    fontFamily:"'Space Mono',monospace"},
  th:{textAlign:"left",padding:"4px 8px",color:"#334155",fontWeight:700,
    letterSpacing:"0.08em",borderBottom:"1px solid #131c28"},
  td:{padding:"5px 8px",color:"#64748b",borderBottom:"1px solid #0d1117",
    verticalAlign:"top",wordBreak:"break-word"},

  errorCard:{background:"#0a0e14",borderRadius:12,border:"1px solid #2a1a1a",padding:"20px 22px"},
  errorHeader:{display:"flex",alignItems:"center",gap:10,marginBottom:8},
  errorX:{width:22,height:22,borderRadius:"50%",background:"#3a1a1a",color:"#ef4444",
    display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,fontWeight:700,flexShrink:0},
  errorTitle:{fontFamily:"'Space Mono',monospace",fontSize:13,fontWeight:700,color:"#ef4444",letterSpacing:"0.1em"},
  errorMsg:{fontSize:13,color:"#64748b",margin:"0 0 4px"},
  tip:{fontSize:12,color:"#4a5568",background:"#0d1117",borderRadius:6,padding:"10px 12px",marginTop:10,lineHeight:1.6},

  histSection:{marginTop:8,border:"1px solid #131c28",borderRadius:10,overflow:"hidden"},
  histHeader:{display:"flex",justifyContent:"space-between",alignItems:"center",
    padding:"10px 16px",borderBottom:"1px solid #131c28",background:"#0a0e14"},
  histLabel:{fontFamily:"'Space Mono',monospace",fontSize:10,color:"#334155",letterSpacing:"0.14em"},
  histClear:{background:"none",border:"none",fontFamily:"'Space Mono',monospace",
    fontSize:9,color:"#2a3a4a",letterSpacing:"0.1em",cursor:"pointer"},
  histList:{background:"#080b0f"},
  histItem:{display:"flex",alignItems:"center",gap:14,padding:"10px 16px",
    borderBottom:"1px solid #0d1117",cursor:"pointer",transition:"background .15s",background:"transparent"},
  histThumb:{width:44,height:44,borderRadius:5,objectFit:"cover",flexShrink:0,border:"1px solid #1a2332"},
  histInfo:{flex:1,minWidth:0},
  histName:{fontSize:13,fontWeight:600,color:"#94a3b8",fontFamily:"'Space Mono',monospace",
    overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},
  histMeta:{display:"flex",alignItems:"center",gap:10,marginTop:3},
  histCat:{fontSize:11,color:"#334155"},
  histSrc:{fontFamily:"'Space Mono',monospace",fontSize:10},
};
