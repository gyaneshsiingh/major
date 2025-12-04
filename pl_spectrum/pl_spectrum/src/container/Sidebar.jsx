import React from "react";
import "../css/Sidebar.css"
const Sidebar = () => {
    return (
    <div className="menu">
        <h2 className="Title">MENU</h2> 
        <ul className="List">
            <li>Upload Video</li>
            <li>Result</li>
            <li>DownLoad Report</li>
        </ul>
    </div>
    )
}

export default Sidebar;