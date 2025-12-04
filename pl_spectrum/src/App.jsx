import React from "react";
import "../src/App.css"
import Sidebar from "./container/Sidebar";
import Video from "./container/Video";
import Result from "./pages/Result";
import Reports from "./pages/Reports";
import { Routes,Route } from "react-router-dom";

const App = () => {
return (
  <div className="app">

  <Sidebar></Sidebar>
    <Routes>
     <Route path="/" element={<Video />} />
          <Route path="/result" element={<Result />} />
          <Route path="/report" element={<Reports />} />
    </Routes>
  </div>

)
}

export default App;