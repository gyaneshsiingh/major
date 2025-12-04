// const express = require('express');
// const cors = require('cors');
// const multer = require("multer");
// const axios = require('axios');
// const fs = require('fs');
// const path = require('path');
// const FormData = require("form-data");


// const app = express();
// app.use(cors());
// app.use(express.json());


// const upload = multer({dest:"uploads/"});

// app.post("/upload",upload.single("video"),async(req,res) => {

//     if (!req.file) {
//         return res.status(400).json({error:"No file uploaded"});
//     }
//     const filepath = req.file.path;
//     const originalname = req.file.originalname;



//     try {
//         const form = new FormData();
//         form.append("video",fs.createReadStream(filepath),originalname);

//         if (req.body.sample_name)  {
//             form.append("sample_name", req.body.sample_name);
//         }
//             const modelResponse = await axios.post("http://127.0.0.1:5003/predict", 
//             form,
//             {headers: form.getHeaders(),
//       maxContentLength: Infinity,
//       maxBodyLength: Infinity,
//       timeout: 5 * 60 * 1000
//     });
//         res.json(modelResponse.data);
//     } catch(err) {
//         console.error(err);
//         res.status(500).json({error:"Failed to process"});

//     }finally {
//         try {
//         fs.unlinkSync(filepath);
//         } catch (e) {}
//     }
// });

// app.listen(5002, () => {
//     console.log("node server is connected to 5002")
// })

const express = require('express');
const cors = require('cors');
const multer = require("multer");
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require("form-data");
const winston = require('winston');

// ============================================================
// Configure Winston Logger
// ============================================================
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        winston.format.printf(({ timestamp, level, message, stack }) => {
            return `${timestamp} - ${level.toUpperCase()} - ${stack || message}`;
        })
    ),
    transports: [
        // Write to console
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.printf(({ timestamp, level, message }) => {
                    return `${timestamp} - ${level} - ${message}`;
                })
            )
        }),
        // Write to file
        new winston.transports.File({ 
            filename: 'server.log',
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.json()
            )
        }),
        // Separate file for errors
        new winston.transports.File({ 
            filename: 'error.log', 
            level: 'error',
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.json()
            )
        })
    ]
});

const app = express();
app.use(cors());
app.use(express.json());

// ============================================================
// Request Logging Middleware
// ============================================================
app.use((req, res, next) => {
    const start = Date.now();
    
    // Log when response finishes
    res.on('finish', () => {
        const duration = Date.now() - start;
        logger.info(`${req.method} ${req.path} - Status: ${res.statusCode} - ${duration}ms`);
    });
    
    next();
});

const upload = multer({ dest: "uploads/" });

// ============================================================
// Upload Endpoint with Detailed Logging
// ============================================================
app.post("/upload", upload.single("video"), async (req, res) => {
    const requestId = `REQ-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    logger.info("============================================================");
    logger.info(`[${requestId}] NEW UPLOAD REQUEST`);
    logger.info("============================================================");

    if (!req.file) {
        logger.warn(`[${requestId}] No file uploaded in request`);
        return res.status(400).json({ error: "No file uploaded" });
    }

    const filepath = req.file.path;
    const originalname = req.file.originalname;
    const filesize = req.file.size;
    const sample_name = req.body.sample_name || "Unknown";

    logger.info(`[${requestId}] File Details:`);
    logger.info(`[${requestId}]   Original Name: ${originalname}`);
    logger.info(`[${requestId}]   Temp Path: ${filepath}`);
    logger.info(`[${requestId}]   Size: ${(filesize / (1024 * 1024)).toFixed(2)} MB`);
    logger.info(`[${requestId}]   Sample Name: ${sample_name}`);

    try {
        // Check if file exists
        if (!fs.existsSync(filepath)) {
            logger.error(`[${requestId}] File not found at path: ${filepath}`);
            return res.status(500).json({ error: "File upload failed" });
        }

        logger.info(`[${requestId}] Creating FormData for ML service...`);
        const form = new FormData();
        form.append("video", fs.createReadStream(filepath), originalname);
        
        if (req.body.sample_name) {
            form.append("sample_name", req.body.sample_name);
        }

        logger.info(`[${requestId}] Sending request to ML service (http://127.0.0.1:5003/predict)...`);
        const startTime = Date.now();

        const modelResponse = await axios.post(
            "http://127.0.0.1:5003/predict",
            form,
            {
                headers: form.getHeaders(),
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                timeout: 5 * 60 * 1000
            }
        );

        const processingTime = Date.now() - startTime;
        logger.info(`[${requestId}] ✓ ML service responded successfully in ${(processingTime / 1000).toFixed(2)}s`);
        
        if (modelResponse.data) {
            logger.info(`[${requestId}] Response Data Summary:`);
            if (modelResponse.data.avg_nm) {
                logger.info(`[${requestId}]   Average Wavelength: ${modelResponse.data.avg_nm.toFixed(2)} nm`);
                logger.info(`[${requestId}]   Peak Wavelength: ${modelResponse.data.peak_nm.toFixed(2)} nm`);
                logger.info(`[${requestId}]   Min Wavelength: ${modelResponse.data.min_nm.toFixed(2)} nm`);
                logger.info(`[${requestId}]   Max Wavelength: ${modelResponse.data.max_nm.toFixed(2)} nm`);
            }
            if (modelResponse.data.processed_frames) {
                logger.info(`[${requestId}]   Processed Frames: ${modelResponse.data.processed_frames}/${modelResponse.data.total_frames}`);
            }
        }

        logger.info(`[${requestId}] ✓✓✓ Request completed successfully ✓✓✓`);
        logger.info("============================================================\n");
        
        res.json(modelResponse.data);

    } catch (err) {
        logger.error(`[${requestId}] ✗✗✗ ERROR processing request ✗✗✗`);
        
        if (err.response) {
            // ML service responded with error
            logger.error(`[${requestId}] ML Service Error Response:`);
            logger.error(`[${requestId}]   Status: ${err.response.status}`);
            logger.error(`[${requestId}]   Status Text: ${err.response.statusText}`);
            logger.error(`[${requestId}]   Data: ${JSON.stringify(err.response.data)}`);
            
            return res.status(err.response.status).json({
                error: "ML service error",
                details: err.response.data
            });
        } else if (err.request) {
            // Request made but no response
            logger.error(`[${requestId}] No response from ML service`);
            logger.error(`[${requestId}]   Error: ${err.message}`);
            
            return res.status(503).json({
                error: "ML service unavailable",
                details: "Could not connect to ML service at http://127.0.0.1:5003"
            });
        } else {
            // Error in request setup
            logger.error(`[${requestId}] Request setup error: ${err.message}`);
            logger.error(`[${requestId}] Stack: ${err.stack}`);
            
            return res.status(500).json({
                error: "Failed to process",
                details: err.message
            });
        }

    } finally {
        // Cleanup uploaded file
        try {
            if (fs.existsSync(filepath)) {
                fs.unlinkSync(filepath);
                logger.info(`[${requestId}] ✓ Cleaned up temp file: ${filepath}`);
            }
        } catch (e) {
            logger.warn(`[${requestId}] ⚠ Failed to delete temp file: ${e.message}`);
        }
        
        logger.info("============================================================\n");
    }
});

// ============================================================
// Health Check Endpoint
// ============================================================
app.get("/health", (req, res) => {
    logger.info("Health check requested");
    
    const health = {
        status: "healthy",
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        uploads_dir_exists: fs.existsSync("uploads")
    };
    
    res.json(health);
});

// ============================================================
// Error Handler Middleware
// ============================================================
app.use((err, req, res, next) => {
    logger.error(`Unhandled error: ${err.message}`);
    logger.error(`Stack: ${err.stack}`);
    res.status(500).json({ error: "Internal server error" });
});

// ============================================================
// Start Server
// ============================================================
const PORT = 5002;

app.listen(PORT, () => {
    logger.info("============================================================");
    logger.info("NODE.JS SERVER STARTING");
    logger.info("============================================================");
    logger.info(`Server running on: http://127.0.0.1:${PORT}`);
    logger.info(`Logs: server.log, error.log`);
    logger.info("Endpoints:");
    logger.info(`  POST   /upload  - Upload and process video`);
    logger.info(`  GET    /health  - Health check`);
    logger.info("============================================================");
    
    // Check if uploads directory exists
    if (!fs.existsSync("uploads")) {
        logger.info("Creating uploads directory...");
        fs.mkdirSync("uploads");
        logger.info("✓ Uploads directory created");
    }
    
    logger.info("✓ Server ready to accept requests\n");
});