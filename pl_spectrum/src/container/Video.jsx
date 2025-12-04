import { useState } from "react";
import axios from "axios";
import Result from "../pages/Result";
import "../css/Video.css"


const Video = () => {
   const[video,setVideo] = useState(null);
   const[sample,setSample] = useState("");
   const[file,setFile] = useState(null);
   const [loading, setLoading] = useState(false);
   const [result,setResult] = useState(null);

   const handleVideo = (e) => {
    const files = e.target.files[0];

    if (files && files.type.startsWith("video/")){
        setVideo(URL.createObjectURL(files));
        setFile(files);
    }
    else {
      alert("Please select a valid video file");
    }
   };

   const handleSubmit = async() => {
     if (!sample || !video) {
      alert("Please enter a sample name and upload a video!");
      return;
    }
    const formData = new FormData();
    formData.append("video", file);
    formData.append("sample_name", sample);

    try {
      setLoading(true);
      const res = await axios.post("http://127.0.0.1:5002/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          timeout: 5 * 60 * 1000
        },
      });
      setResult(res.data);
      alert(" Video uploaded successfully!");
    } catch (error) {
      console.error("Error uploading video:", error);
      alert(" Something went wrong while uploading.");
    } finally {
      setLoading(false);
    }
  };

    return (
        <>
        <div className="main"> 
            <div className="upload-card">
            <h1 className="title_video">Upload Video</h1>
            <input type="text" className="sample-input"  placeholder="Enter Sample name" value ={sample} onChange={(e) => setSample(e.target.value)}/>
            <label htmlFor="videoInput" className="upload-box">
            <span>🎬 Click to select or drag your video here</span>
            <input type="file" id = "videoInput" accept="video/*" onChange={handleVideo} hidden/>
            </label>
            {video && (<video 
            src = {video}
            controls
            width="100%"
             style={{ borderRadius: "10px", marginTop: "15px" }}
        />
      )}
      <button  className="upload-btn" onClick={handleSubmit}>Upload</button>
       {result && <Result result={result} />}
      </div>
      </div>
          </>
    )
}

export default Video;