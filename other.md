# Trinetra VMS Backend Architecture Split - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture Requirements](#architecture-requirements)
3. [Current System Analysis](#current-system-analysis)
4. [Split Architecture Design](#split-architecture-design)
5. [Implementation Details](#implementation-details)
6. [File Structure Changes](#file-structure-changes)
7. [Communication Flow](#communication-flow)
8. [Deployment Strategy](#deployment-strategy)
9. [Security Considerations](#security-considerations)
10. [Next Steps](#next-steps)
11. [Migration Guide](#migration-guide)

---

## Overview

This document outlines the complete restructuring of the Trinetra VMS (Video Management System) backend from a monolithic architecture to a hybrid cloud/on-premise split architecture. The changes were implemented to meet specific requirements for cloud scalability while maintaining local processing capabilities.

### Date: July 22, 2025
### Project: Trinetra VMS Backend Split Architecture
### Architecture Type: Hybrid Cloud/On-Premise

---

## Architecture Requirements

Based on the architecture context provided, the system requires:

### Core Requirements
- **Frontend**: Entirely deployed on DigitalOcean (cloud)
- **Backend Split**: Two-part backend system
  - Cloud backend on DigitalOcean (communicates with frontend)
  - On-premise backend on client PC (handles local operations)
- **Communication**: HTTP/REST API between backend parts
- **Data Sovereignty**: Local processing for sensitive operations
- **Performance**: Local handling of video streams and analytics
- **Offline Operation**: On-premise operations continue without cloud connectivity

### Technical Requirements
- **Protocol**: HTTP/HTTPS using RESTful APIs
- **Authentication**: JWT, API keys, or OAuth
- **Security**: CORS configuration, secure tunnels/VPN for cloud-to-site calls
- **Technologies**: Node.js Express for both backends

---

## Current System Analysis

### Original Structure Analysis
The original `trinetra-vms-be` folder contained:

```
trinetra-vms-be/
├── index.js                    # Main server entry
├── package.json               # Dependencies
├── config/                    # Database and server config
├── controller/                # Business logic controllers
│   ├── auth/                  # Authentication
│   ├── user/                  # User management
│   ├── camera/                # Camera operations
│   ├── stream/                # Video streaming
│   ├── aiAnalytics/           # AI processing
│   └── aiModels/              # AI model management
├── routes/                    # API endpoints
├── middlewarer/               # Custom middleware
├── RecordingCode/             # Video recording functionality
├── helpers/                   # Utility functions
├── customVariables/           # Custom error definitions
└── scripts/                   # Database and setup scripts
```

### Key Components Identified
1. **Cloud-suitable components**: User auth, user management, API routing
2. **On-premise components**: Camera operations, video streaming, AI analytics, recording
3. **Shared components**: Utilities, models, custom variables

---

## Split Architecture Design

### Design Principles
1. **Separation of Concerns**: Cloud handles user-facing operations, on-prem handles hardware
2. **API-First**: All communication through well-defined REST APIs
3. **Security**: Secure authentication between components
4. **Scalability**: Cloud components can scale independently
5. **Reliability**: On-prem operations continue during cloud outages

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD (DigitalOcean)                     │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React/Vue/Angular)                              │
│           │                                                 │
│           ▼                                                 │
│  Cloud Backend (Node.js Express)                           │
│  - User Authentication                                      │
│  - User Management                                          │
│  - API Proxy to On-Prem                                    │
│  - Frontend Communication                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/HTTPS REST API
                      │ (Secure Communication)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 ON-PREMISE (Client Site)                    │
├─────────────────────────────────────────────────────────────┤
│  On-Premise Backend (Node.js Express)                      │
│  - Camera Management                                        │
│  - Video Streaming                                          │
│  - AI Analytics                                             │
│  - Video Recording                                          │
│  - Local Device Communication                               │
│           │                                                 │
│           ▼                                                 │
│  Local Infrastructure                                       │
│  - IP Cameras                                               │
│  - Local Storage                                            │
│  - AI Processing Hardware                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Directory Structure Created

#### 1. Cloud Backend (`/cloud/`)
**Purpose**: Handles frontend communication and user management
**Port**: 3001 (default)

```
cloud/
├── index.js                   # Main server file with Express setup
├── package.json              # Cloud-specific dependencies
├── .env.example              # Environment variables template
├── routes/
│   ├── auth.js               # Authentication endpoints
│   ├── user.js               # User management endpoints
│   └── proxy.js              # Proxy routes to on-prem backend
├── controller/
│   ├── auth/                 # Authentication controllers
│   └── user/                 # User management controllers
├── config/                   # Cloud configuration files
└── middlewarer/              # Cloud middleware (auth, CORS, etc.)
```

**Key Dependencies**:
- express: Web framework
- cors: Cross-origin resource sharing
- jsonwebtoken: JWT authentication
- axios: HTTP client for on-prem communication
- helmet: Security middleware
- express-rate-limit: Rate limiting

#### 2. On-Premise Backend (`/onprem/`)
**Purpose**: Handles local hardware operations and processing
**Port**: 3002 (default)

```
onprem/
├── index.js                  # Main server file
├── package.json             # On-prem specific dependencies
├── .env.example             # Environment variables template
├── routes/
│   ├── camera.js            # Camera management endpoints
│   ├── stream.js            # Video streaming endpoints
│   ├── analytics.js         # AI analytics endpoints
│   └── recording.js         # Recording management endpoints
├── controller/
│   ├── camera/              # Camera operation controllers
│   ├── stream/              # Streaming controllers
│   └── aiAnalytics/         # AI processing controllers
├── config/                  # Local configuration
├── middlewarer/             # Local middleware
└── RecordingCode/           # Video recording functionality
```

**Key Dependencies**:
- express: Web framework
- fluent-ffmpeg: Video processing
- socket.io: Real-time communication
- knex: Database query builder
- node-media-server: Media streaming
- sqlite3/mysql2: Local database

#### 3. Shared Components (`/shared/`)
**Purpose**: Common utilities and models for both backends

```
shared/
├── package.json             # Shared dependencies
├── utils/                   # Utility functions
├── models/                  # Data models
├── customVariables/         # Custom error definitions
└── helpers/                 # Helper functions
```

### Configuration Files Created

#### Cloud Backend Environment (`.env.example`)
```env
NODE_ENV=development
PORT=3001
FRONTEND_URL=http://localhost:3000
ONPREM_BACKEND_URL=http://localhost:3002
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trinetra_vms_cloud
JWT_SECRET=your-super-secret-jwt-key
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

#### On-Premise Backend Environment (`.env.example`)
```env
NODE_ENV=development
PORT=3002
CLOUD_BACKEND_URL=http://localhost:3001
DB_HOST=localhost
DB_PORT=3306
DB_NAME=trinetra_vms_local
DEFAULT_RTSP_PORT=554
RECORDING_PATH=./recordings
CHUNK_DURATION=10
AI_MODELS_PATH=./models
ANALYTICS_OUTPUT_PATH=./analytics
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=1000
```

---

## Communication Flow

### 1. User Authentication Flow
```
Frontend → Cloud Backend (/api/auth/login)
Cloud Backend → Database (verify credentials)
Cloud Backend → Frontend (JWT token)
```

### 2. Camera Operations Flow
```
Frontend → Cloud Backend (/api/proxy/cameras)
Cloud Backend → On-Prem Backend (/api/cameras)
On-Prem Backend → Local Cameras (RTSP/HTTP)
On-Prem Backend → Cloud Backend (response)
Cloud Backend → Frontend (proxied response)
```

### 3. Video Streaming Flow
```
Frontend → Cloud Backend (/api/proxy/streams/:id/live)
Cloud Backend → On-Prem Backend (/api/streams/:id/live)
On-Prem Backend → Camera (RTSP stream)
On-Prem Backend → Frontend (stream URL/WebRTC)
```

### 4. AI Analytics Flow
```
Frontend → Cloud Backend (/api/proxy/analytics/start)
Cloud Backend → On-Prem Backend (/api/analytics/start)
On-Prem Backend → AI Processing (local models)
On-Prem Backend → Local Database (results)
Results available via polling or WebSocket
```

---

## Deployment Strategy

### Cloud Deployment (DigitalOcean)

#### Option 1: App Platform
```bash
# Build and deploy via App Platform
npm run build
# Configure App Platform with GitHub integration
# Set environment variables in App Platform dashboard
```

#### Option 2: Droplet with Docker
```bash
# Create Dockerfile for cloud backend
docker build -t trinetra-vms-cloud .
docker run -p 3001:3001 trinetra-vms-cloud
```

### On-Premise Deployment

#### Installation Package
```bash
# Create installation script
#!/bin/bash
# Install Node.js
# Install FFmpeg
# Copy application files
# Install dependencies
npm install
# Create systemd service
# Start service
```

#### Service Configuration
```ini
[Unit]
Description=Trinetra VMS On-Premise Backend
After=network.target

[Service]
Type=simple
User=trinetra
WorkingDirectory=/opt/trinetra-vms
ExecStart=/usr/bin/node index.js
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Security Considerations

### Authentication & Authorization
1. **JWT Tokens**: For frontend to cloud backend communication
2. **API Keys**: For cloud to on-premise backend communication
3. **Rate Limiting**: Implemented on both backends
4. **CORS Configuration**: Properly configured for frontend access

### Network Security
1. **HTTPS**: All communication must use HTTPS in production
2. **VPN/Tunneling**: For secure cloud-to-site communication
3. **Firewall Rules**: Restrict on-premise backend access
4. **API Versioning**: Implement versioned APIs for compatibility

### Data Security
1. **Data Encryption**: Encrypt sensitive data in transit and at rest
2. **Local Storage**: Keep sensitive video data on-premise
3. **Audit Logging**: Log all API calls and access attempts
4. **Backup Strategy**: Separate backup strategies for cloud and on-prem data

---

## Next Steps

### Immediate Actions (Priority 1)
1. **Code Migration**: Move existing controllers and routes to appropriate backends
   - Auth/User controllers → Cloud backend
   - Camera/Stream/Analytics controllers → On-premise backend
   - Shared utilities → Shared folder

2. **Environment Setup**: Configure environment variables for both backends
3. **Database Configuration**: Set up separate databases for cloud and on-prem
4. **Local Testing**: Test communication between both backends

### Short-term Actions (Priority 2)
5. **Security Implementation**: Add authentication middleware and API keys
6. **Error Handling**: Implement proper error handling and logging
7. **API Documentation**: Create API documentation for both backends
8. **Unit Testing**: Write tests for both backend components

### Medium-term Actions (Priority 3)
9. **Cloud Deployment**: Deploy cloud backend to DigitalOcean
10. **Client Packaging**: Create installation package for on-premise backend
11. **Monitoring**: Set up monitoring and alerting for both backends
12. **Performance Optimization**: Optimize communication and processing

### Long-term Actions (Priority 4)
13. **Scaling**: Implement horizontal scaling for cloud backend
14. **Advanced Features**: Add advanced analytics and reporting
15. **Mobile App**: Extend API for mobile application support
16. **Multi-tenant**: Support multiple client deployments

---

## Migration Guide

### Step 1: Backup Current System
```bash
# Create backup of current backend
cp -r trinetra-vms-be trinetra-vms-be-backup
```

### Step 2: Install Dependencies
```bash
# Cloud backend
cd cloud && npm install

# On-premise backend
cd onprem && npm install

# Shared components
cd shared && npm install
```

### Step 3: Move Authentication Code
```bash
# Move auth controllers
mv ../controller/auth/* ./cloud/controller/auth/
mv ../routes/auth.js ./cloud/routes/auth.js

# Move user controllers
mv ../controller/user/* ./cloud/controller/user/
mv ../routes/user.js ./cloud/routes/user.js
```

### Step 4: Move Camera/Stream Code
```bash
# Move camera controllers
mv ../controller/camera/* ./onprem/controller/camera/
mv ../routes/camera.js ./onprem/routes/camera.js

# Move stream controllers
mv ../controller/stream/* ./onprem/controller/stream/
mv ../routes/stream.js ./onprem/routes/stream.js

# Move analytics controllers
mv ../controller/aiAnalytics/* ./onprem/controller/aiAnalytics/
mv ../routes/aiAnalytics.js ./onprem/routes/analytics.js
```

### Step 5: Move Shared Components
```bash
# Move helpers and utilities
mv ../helpers/* ./shared/helpers/
mv ../customVariables/* ./shared/customVariables/
```

### Step 6: Update Import Paths
Update all require() statements to use relative paths for the new structure.

### Step 7: Configure Databases
Set up separate database connections for cloud and on-premise backends.

### Step 8: Test Locally
```bash
# Start cloud backend
cd cloud && npm run dev

# Start on-premise backend (in another terminal)
cd onprem && npm run dev

# Test API endpoints
curl http://localhost:3001/health
curl http://localhost:3002/health
```

---

## Code Examples

### Cloud Backend Proxy Implementation
```javascript
// cloud/routes/proxy.js
router.use('/cameras/*', async (req, res) => {
  try {
    const response = await axios({
      method: req.method,
      url: `${ONPREM_BACKEND_URL}/api${req.originalUrl.replace('/api/proxy', '')}`,
      data: req.body,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': req.headers.authorization
      }
    });
    res.status(response.status).json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to communicate with on-premise backend' });
  }
});
```

### On-Premise Authentication Middleware
```javascript
// onprem/middlewarer/auth.js
const verifyCloudAuth = async (req, res, next) => {
  try {
    const token = req.headers.authorization;
    const response = await axios.get(`${CLOUD_BACKEND_URL}/api/auth/verify`, {
      headers: { authorization: token }
    });
    req.user = response.data.user;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Unauthorized' });
  }
};
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Between Backends
**Problem**: Cloud backend cannot reach on-premise backend
**Solution**: 
- Check network connectivity
- Verify firewall rules
- Ensure on-premise backend is running
- Check environment variables

#### 2. CORS Issues
**Problem**: Frontend cannot access cloud backend
**Solution**:
- Update CORS configuration in cloud backend
- Verify FRONTEND_URL environment variable
- Check browser console for CORS errors

#### 3. Authentication Failures
**Problem**: Authentication between backends fails
**Solution**:
- Verify JWT_SECRET consistency
- Check token expiration
- Validate API key configuration

---

## Conclusion

This documentation provides a complete guide for the Trinetra VMS backend architecture split. The new structure enables:

- **Scalable cloud deployment** for user-facing operations
- **Local processing** for sensitive video operations
- **Secure communication** between components
- **Independent scaling** of cloud and on-premise components
- **Offline operation** capability for critical functions

The implementation follows best practices for microservices architecture while maintaining the simplicity needed for effective deployment and maintenance.

---

**Document Version**: 1.0  
**Last Updated**: July 22, 2025  
**Created By**: AI Assistant  
**Review Status**: Initial Draft
