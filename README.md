<img width="1536" height="1024" alt="8b21cf02-38b5-4b0e-bf33-d52cdb5cf8ff" src="https://github.com/user-attachments/assets/00b48c11-13d9-434c-81ab-37e267a613a1" />

# EquiFall+  
### AI-Powered Fall Detection and Emergency Response System

Turning passive cameras into active caregivers.
<img width="1536" height="1024" alt="equifall" src="https://github.com/user-attachments/assets/d9e66c97-fd53-4f96-a438-fd4895506bcc" />

---

## One-Line Pitch
EquiFall+ is an AI-powered system that detects falls in real time using video input and instantly alerts caregivers to ensure rapid response and improved elderly safety.

---

## Problem
Falls are one of the leading causes of injury among elderly individuals, especially those living alone.

- Delayed response can be life-threatening  
- Lack of real-time monitoring  
- Existing solutions rely on wearables, which are often ignored or forgotten  

---

## Solution
EquiFall+ uses computer vision to detect falls instantly from:
- CCTV cameras  
- Webcam feeds  
- Uploaded videos  

It then:
- Sends alerts automatically  
- Stores incidents  
- Provides analytics for prevention  

---

## Features

- Real-time fall detection  
- Pose-based AI analysis using YOLOv8  
- Emergency alerts via Telegram, SMS, and Email  
- Incident history and replay  
- Monitoring dashboard  
- Voice-based safety check (local environments only)  

---

## Tech Stack

### Frontend
- Lovable (React-based UI)

### Backend
- FastAPI (Python)

### AI / Computer Vision
- YOLOv8 Pose Model  
- OpenCV  

### Deployment
- Render (Backend)  
- Lovable (Frontend)

---

## How It Works

1. Capture video input (CCTV, Webcam, or Upload)  
2. Process frames using AI model  
3. Detect posture and sudden fall movement  
4. Trigger safety check  
5. If no response, notify emergency contacts  
6. Log incident and generate insights  

---

## Architecture

```mermaid
        flowchart TD
        A[Camera / Upload] --> B[Fall Detection Model using YOLOv8] --> C[FastAPI Backend] --> D[Alert System with Telegram / SMS / Email] --> E[Dashboard and Analytics];
        

```
    
## The Lovable Website

### Working:

https://github.com/user-attachments/assets/2023fe13-3dbf-46c8-a6a2-bf1d00cce7b1


### Fall detected bounding box:
<img width="1284" height="640" alt="image" src="https://github.com/user-attachments/assets/9c794622-f2c3-455c-abbc-e3df2c12e2fa" />

### Protocol to assess the situation:
<img width="1600" height="813" alt="h2" src="https://github.com/user-attachments/assets/5532862d-29eb-4c12-8100-63a46b73cbc2" />

### Which provides this automated notification to emergency contacts:
<img width="1970" height="730" alt="image" src="https://github.com/user-attachments/assets/721adec8-5a8d-4172-beeb-e82d7441fa17" />

### Incident timeline:
<img width="1600" height="380" alt="inci" src="https://github.com/user-attachments/assets/e3fc2388-8456-4b60-962b-63d7d51b58c9" />

### The generated report:
<img width="1600" height="933" alt="report" src="https://github.com/user-attachments/assets/13804e84-b812-4da5-98f3-6f87e49c7101" />


## Installation (Local Setup)

-- git clone https://github.com/arthika333/equifall.git
-- cd equifall
-- pip install -r requirements.txt python app.py
