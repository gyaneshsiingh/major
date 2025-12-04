const express = require('express');
const cors = require('cors');
const multer = require("multer");
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require("form-data");


const app = express();
app.use(cors());
app.use(express.json());


const upload = multer({dest:"uploads/"});

app.post("/upload",upload.single("video"),async(req,res) => {

    if (!req.file) {
        return res.status(400).json({error:"No file uploaded"});
    }
    const filepath = req.file.path;
    const originalname = req.file.originalname;



    try {
        const form = new FormData();
        form.append("video",fs.createReadStream(filepath),originalname);

        if (req.body.sample_name)  {
            form.append("sample_name", req.body.sample_name);
        }
            const modelResponse = await axios.post("http://127.0.0.1:5003/predict", 
            form,
            {headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 5 * 60 * 1000
    });
        res.json(modelResponse.data);
    } catch(err) {
        console.error(err);
        res.status(500).json({error:"Failed to process"});

    }finally {
        try {
        fs.unlinkSync(filepath);
        } catch (e) {}
    }
});

app.listen(5002, () => {
    console.log("node server is connected to 5002")
})