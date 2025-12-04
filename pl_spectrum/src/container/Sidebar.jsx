import React from "react";
import "../css/Sidebar.css"
import { NavLink } from "react-router-dom";
const Sidebar = () => {
    return (
    <div className="menu">
        <h1 className="Dash">AUTOPSY DASHBOARD</h1>
        <h2 className="Title">MENU</h2> 
        <ul className="List">
            <li><NavLink to = "/" className="nav-link">Upload Video</NavLink></li>
            <li><NavLink to = "/result" className="nav-link">Result</NavLink></li>
            <li><NavLink to = "/report" className="nav-link">Download Report</NavLink></li>
        </ul>
    </div>
    )
}

export default Sidebar;