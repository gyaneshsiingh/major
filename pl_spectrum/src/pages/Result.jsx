import React from "react";
import "../css/Result.css"
const Result = ({result}) => {
    if (!result) return null;

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
        const rows = [["wavelength_nm","intensity"]];
        for (let i = 0; i < wavelength_list.length; i++) {
            rows.push([wavelength_list[i],intensity_list[i] ?? ""]);
        }

        const csvContent = rows.map(e => e.join(",")).join("\n");
        const blob = new Blob([csvContent], {type: "text/csv"});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${sample_name || "sample"}_wavelength.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="result-container">
            <h2 className="result-title">Results: {sample_name || "Sample"}</h2>
            <div className="result-content">
                <div className="result-details">
                    <p><strong>Average Wavelength:</strong>{Number(avg_nm).toFixed(2)}</p>
                    <p><strong>Peak Wavelength:</strong>{Number(peak_nm).toFixed(2)}</p>
                    <p><strong>Min Wavelength:</strong>{Number(min_nm).toFixed(2)}</p>
                    <p><strong>Max Wavelength:</strong>{Number(max_nm).toFixed(2)}</p>
                </div>

                <div className="result-plot">
                    {plot_base64 ? (
                        <img className="result-image" src = {plot_base64} alt = "Wavelength vs Intensity" />
                    ):(<p>No Plot Returned. </p>)
                    }
                </div>
            </div>

            <div className="result-actions">
                <button className="csv-btn" onClick={downloadCSV}>Download CSV</button>


                <details className="result-raw">
                    <summary>Raw Lists (Preview)</summary>

                    <div>
                        <pre>
                            wavelengths: {JSON.stringify(wavelength_list.slice(0,200),null, 2)}
                            {"\n"}
                            intensities: {JSON.stringify(intensity_list.slice(0,200),null, 2)}
                        </pre>
                    </div>
                </details>
            </div>
        </div>
    );
};

export default Result;