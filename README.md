# SafeShopAI
SentinelEye is a deep learning-powered solution for real-time theft detection in retail surveillance videos.
---
## **Visual Demo**

### **System Logs**
![System Logs](Assets/SystemLogs.png)

### **Upload & Record Interface**
![Upload & Record](Assets/UploadRecord.png)

### **Demo Video**
[▶ Watch Demo Video](Assets/demo_video.mp4)

---

## **📂 Project Structure**
```
SafeShopAI/
│
├── data/                     # dataset placeholder (not tracked in git)
├── notebooks/                # Jupyter notebooks for exploration
│   └── video_preprocessing.ipynb
│
├── src/                      # core source code
│   ├── models/               # model architectures (custom & pretrained)
│   ├── preprocessing/        # frame extraction, augmentation
│   ├── training/             # training scripts & loops
│   ├── inference/            # inference pipeline
│   ├── utils/                # helper functions (logging, metrics)
│   └── config.py             # project-wide configs
│
├── tests/                    # unit & integration tests
│
├── docker/                   # Docker-related files
│   └── Dockerfile
│
├── .github/workflows/        # GitHub Actions CI/CD
│   └── ci.yml
│
├── requirements.txt          # project dependencies
├── README.md                 # main documentation
├── .gitignore                # files to ignore
└── setup.py                  # packaging (optional)
```
