import React, {useState,useEffect} from "react";
import axios from "axios";


const SavedResult = () => {
    const [data, setData] = useState(null);


    useEffect(() => {
        axios.get("http://127.0.0.1:5003/saved-results")
        .then(res => setData(res.data))
        .catch(err => console.error(err));
    },[]);

    return (
        <div className="main">
            <div className="upload-card">
                <h1>Saved Result History</h1>

                {data.length === 0 && <p>No saved data found</p>}

                {data.map((item,index) => (
                    <div key = {index} className="history-items">
                        <h3>{item.sample_name}</h3>
                        <p>{item.timestamp}</p>
                
                        <img src = {item.plot_base64} alt = "plot" style = {{width:"300px"}}/>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SavedResult;