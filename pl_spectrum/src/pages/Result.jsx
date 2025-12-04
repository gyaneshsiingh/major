// import React from "react";
// import "../css/Result.css"
// const Result = ({result}) => {
//     if (!result) return null;

//     const {
//         sample_name,
//         avg_nm,
//         peak_nm,
//         min_nm,
//         max_nm,
//         wavelength_list = [],
//         intensity_list = [],
//         plot_base64
//     } = result;

//     const downloadCSV = () => {
//         const rows = [["wavelength_nm","intensity"]];
//         for (let i = 0; i < wavelength_list.length; i++) {
//             rows.push([wavelength_list[i],intensity_list[i] ?? ""]);
//         }

//         const csvContent = rows.map(e => e.join(",")).join("\n");
//         const blob = new Blob([csvContent], {type: "text/csv"});
//         const url = URL.createObjectURL(blob);
//         const a = document.createElement("a");
//         a.href = url;
//         a.download = `${sample_name || "sample"}_wavelength.csv`;
//         a.click();
//         URL.revokeObjectURL(url);
//     };

//     return (
//         <div className="result-container">
//             <h2 className="result-title">Results: {sample_name || "Sample"}</h2>
//             <div className="result-content">
//                 <div className="result-details">
//                     <p><strong>Average Wavelength:</strong>{Number(avg_nm).toFixed(2)}</p>
//                     <p><strong>Peak Wavelength:</strong>{Number(peak_nm).toFixed(2)}</p>
//                     <p><strong>Min Wavelength:</strong>{Number(min_nm).toFixed(2)}</p>
//                     <p><strong>Max Wavelength:</strong>{Number(max_nm).toFixed(2)}</p>
//                 </div>

//                 <div className="result-plot">
//                     {plot_base64 ? (
//                         <img className="result-image" src = {plot_base64} alt = "Wavelength vs Intensity" />
//                     ):(<p>No Plot Returned. </p>)
//                     }
//                 </div>
//             </div>

//             <div className="result-actions">
//                 <button className="csv-btn" onClick={downloadCSV}>Download CSV</button>


//                 <details className="result-raw">
//                     <summary>Raw Lists (Preview)</summary>

//                     <div>
//                         <pre>
//                             wavelengths: {JSON.stringify(wavelength_list.slice(0,200),null, 2)}
//                             {"\n"}
//                             intensities: {JSON.stringify(intensity_list.slice(0,200),null, 2)}
//                         </pre>
//                     </div>
//                 </details>
//             </div>
//         </div>
//     );
// };

// export default Result;

import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import "../css/Result.css";

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  useEffect(() => {
    // Redirect back if no result data
    if (!result) {
      navigate("/");
    }
  }, [result, navigate]);

  if (!result) {
    return null;
  }

  const {
    sample_name,
    avg_nm,
    peak_nm,
    min_nm,
    max_nm,
    wavelength_list = [],
    intensity_list = [],
    plot_base64
  } = result;

  const downloadCSV = () => {
    const rows = [["wavelength_nm", "intensity"]];
    for (let i = 0; i < wavelength_list.length; i++) {
      rows.push([wavelength_list[i], intensity_list[i] ?? ""]);
    }
    const csvContent = rows.map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sample_name || "sample"}_wavelength.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadImage = () => {
    if (!plot_base64) {
      alert("No plot available to download");
      return;
    }
    const a = document.createElement("a");
    a.href = plot_base64;
    a.download = `${sample_name || "sample"}_plot.png`;
    a.click();
  };

  const downloadAllData = () => {
    // Create a comprehensive JSON file with all data
    const allData = {
      sample_name,
      statistics: {
        avg_nm,
        peak_nm,
        min_nm,
        max_nm
      },
      wavelength_list,
      intensity_list,
      plot_image: plot_base64
    };

    const jsonContent = JSON.stringify(allData, null, 2);
    const blob = new Blob([jsonContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sample_name || "sample"}_complete_data.json`;
    a.click();
    URL.revokeObjectURL(url);

    // Also download CSV and image
    setTimeout(() => {
      downloadCSV();
      setTimeout(() => {
        downloadImage();
      }, 500);
    }, 500);
  };

  return (
    <div className="result-container">
      <div className="result-header">
        <h2 className="result-title">Results: {sample_name || "Sample"}</h2>
        <button onClick={() => navigate("/")} className="back-button">
          ← Upload New Video
        </button>
      </div>

      <div className="result-content">
        <div className="result-details">
          <h3>Wavelength Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Average Wavelength:</span>
              <span className="stat-value">{Number(avg_nm).toFixed(2)} nm</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Peak Wavelength:</span>
              <span className="stat-value">{Number(peak_nm).toFixed(2)} nm</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Min Wavelength:</span>
              <span className="stat-value">{Number(min_nm).toFixed(2)} nm</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Max Wavelength:</span>
              <span className="stat-value">{Number(max_nm).toFixed(2)} nm</span>
            </div>
          </div>
        </div>

        <div className="result-plot">
          <h3>Wavelength vs Intensity Plot</h3>
          {plot_base64 ? (
            <img 
              className="result-image" 
              src={plot_base64} 
              alt="Wavelength vs Intensity" 
            />
          ) : (
            <p className="no-plot">No Plot Returned.</p>
          )}
        </div>
      </div>

      <div className="result-actions">
        <button className="action-btn primary-btn" onClick={downloadAllData}>
          📦 Download All Data
        </button>
        <button className="action-btn csv-btn" onClick={downloadCSV}>
          📊 Download CSV
        </button>
        <button className="action-btn image-btn" onClick={downloadImage}>
          🖼️ Download Plot Image
        </button>
      </div>

      <details className="result-raw">
        <summary>Raw Lists (Preview - First 200 entries)</summary>
        <div className="raw-data">
          <pre>
            <strong>Wavelengths:</strong>
            {JSON.stringify(wavelength_list.slice(0, 200), null, 2)}
            {"\n\n"}
            <strong>Intensities:</strong>
            {JSON.stringify(intensity_list.slice(0, 200), null, 2)}
          </pre>
        </div>
      </details>
    </div>
  );
};

export default Result;