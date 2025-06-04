<div align="center">

![Header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=PhoBERT%20Comment%20Classifier&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Mô%20hình%20phân%20loại%20cảm%20xúc%20bình%20luận%20tiếng%20Việt%20thông%20minh&descAlignY=51&descAlign=62)

</div>

# <div align="center">🚀 PhoBERT Comment Classifier</div>
### <div align="center">*Mô hình phân loại cảm xúc bình luận tiếng Việt thông minh*</div>

<div align="center">

![PhoBERT](https://img.shields.io/badge/Model-PhoBERT-blue?style=for-the-badge&logo=huggingface&logoColor=white)
![Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red?style=for-the-badge&logo=google-translate&logoColor=white)
![AI](https://img.shields.io/badge/AI-NLP-green?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logoColor=black)

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-phobert--vi--comment--4class-ff6b35?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/vanhai123/phobert-vi-comment-4class)
[![Dataset](https://img.shields.io/badge/📊%20Dataset-Vietnamese%20Social%20Comments-purple?style=flat-square&logo=database&logoColor=white)](https://huggingface.co/datasets/vanhai123/vietnamese-social-comments)
[![Demo](https://img.shields.io/badge/🎮%20Demo-Gradio%20App-orange?style=flat-square&logo=streamlit&logoColor=white)](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=36BCF7&center=true&vCenter=true&width=600&lines=🤖+AI+Powered+Vietnamese+NLP;📊+86%25+Accuracy+Performance;🚀+Real-time+Comment+Analysis;⚡+PhoBERT+Base+Architecture)

</div>

---

<div align="center">

![Snake Animation](https://github.com/vanhai123/vanhai123/blob/output/github-contribution-grid-snake-dark.svg)

</div>

## 🎯 **Tổng quan dự án**

> 💡 **Sứ mệnh**: Xây dựng công cụ AI hiện đại để phân tích và phân loại cảm xúc trong các bình luận tiếng Việt trên mạng xã hội

<div align="center">
  
![Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=vanhai123&theme=github-compact&bg_color=0d1117&color=58a6ff&line=f85149&point=ffffff&area=true&hide_border=true)

</div>

<table>
<tr>
<td width="50%">

### 🎭 **Khả năng phân loại**
<div align="center">

```mermaid
flowchart LR
    A[📝 Input Comment] --> B{🤖 PhoBERT}
    B --> C[🟢 Positive]
    B --> D[🔴 Negative]
    B --> E[⚪ Neutral]
    B --> F[⚠️ Toxic]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#ffebee
    style E fill:#fafafa
    style F fill:#fff3e0
```

- 🟢 **Positive** - Tích cực
- 🔴 **Negative** - Tiêu cực  
- ⚪ **Neutral** - Trung lập
- ⚠️ **Toxic** - Kích động, phản cảm

</div>
</td>
<td width="50%">

### 📱 **Nguồn dữ liệu**
<div align="center">

```mermaid
mindmap
  root((🌐 Data Sources))
    🎵 TikTok
      Comments
      Reactions
    📘 Facebook
      Posts
      Reviews
    🎬 YouTube
      Comments
      Reviews
    📊 Others
      Forums
      Social Media
```

- 🎵 TikTok Comments
- 📘 Facebook Posts
- 🎬 YouTube Reviews
- 🌐 Các platform khác

</div>
</td>
</tr>
</table>

---

## 📊 **Thông tin Dataset**

<div align="center">

![Data Visualization](https://github-readme-stats.vercel.app/api?username=vanhai123&show_icons=true&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=58a6ff&icon_color=1f6feb&text_color=c9d1d9)

| 📈 **Metric** | 📋 **Value** | 🎯 **Description** |
|:-------------:|:------------:|:-------------------|
| **📝 Comments** | `4,896` | Tổng số bình luận được thu thập |
| **🏷️ Labels** | `4 classes` | positive, negative, neutral, toxic |
| **🌐 Sources** | `Multi-platform` | TikTok, Facebook, YouTube |
| **📊 Fields** | `3 columns` | comment, label, category |

</div>

<div align="center">

```mermaid
pie title Dataset Distribution
    "🟢 Positive" : 35
    "🔴 Negative" : 25
    "⚪ Neutral" : 25
    "⚠️ Toxic" : 15
```

</div>

<details>
<summary>🔍 <strong>Chi tiết phân bố dữ liệu</strong></summary>

```
📊 Label Distribution:
├── 🟢 Positive: ~35%
├── 🔴 Negative: ~25% 
├── ⚪ Neutral:  ~25%
└── ⚠️ Toxic:    ~15%
```

</details>

---

## ⚡ **Cài đặt nhanh**

### 🛠️ **Requirements**

```bash
# 📦 Cài đặt các thư viện cần thiết
pip install transformers datasets scikit-learn sentencepiece torch
```

<details>
<summary>💻 <strong>Chi tiết dependencies</strong></summary>

```txt
transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.1.0
sentencepiece>=0.1.97
torch>=1.12.0
gradio>=3.0.0  # Cho demo app
```

</details>

---

## 🏗️ **Hướng dẫn Training**

<div align="center">

![Training Process](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=500&color=F77B00&center=true&vCenter=true&width=500&lines=🔧+Initialize+PhoBERT+Model;📊+Load+Vietnamese+Dataset;🏃‍♂️+Training+Process;✅+Model+Evaluation)

</div>

### 🚀 **Quick Start**

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant M as 🤖 Model
    participant D as 📊 Dataset
    participant T as 🏃‍♂️ Trainer
    
    U->>M: Load PhoBERT
    M->>D: Load Vietnamese Data
    D->>T: Start Training
    T->>M: Update Weights
    M->>U: Return Trained Model
    
    Note over U,M: 🎯 86% Accuracy Achieved!
```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔧 Khởi tạo model và tokenizer
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=4
)
```

### 📋 **Training Process**

<table>
<tr>
<td width="50%">

**🎯 Bước 1: Chuẩn bị**
```python
# Load dataset
from datasets import load_dataset
dataset = load_dataset("vanhai123/vietnamese-social-comments")
```

</td>
<td width="50%">

**🏃‍♂️ Bước 2: Training**
```python
# Chạy training script
python train.py
# hoặc sử dụng notebook
jupyter notebook train.ipynb
```

</td>
</tr>
</table>

---

## 📈 **Kết quả Performance**

<div align="center">

![Performance Metrics](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=4000&pause=1000&color=00D084&center=true&vCenter=true&width=600&lines=🎯+86%25+Overall+Accuracy;📊+83%25+Macro+F1+Score;🟢+Best%3A+Positive+Class;⚠️+Strong%3A+Toxic+Detection)

### 🏆 **Model Performance**

| 📊 **Metric** | 📈 **Score** | 🎯 **Details** |
|:-------------:|:------------:|:---------------|
| **🎯 Accuracy** | `~86%` | Độ chính xác tổng thể |
| **📊 Macro F1** | `~83%` | F1-score trung bình |
| **🟢 Best Class** | `Positive` | Phân loại tốt nhất |
| **⚠️ Strong Class** | `Toxic` | Nhận diện tốt nội dung độc hại |

</div>

<div align="center">

```mermaid
gantt
    title 📊 Model Performance Timeline
    dateFormat  X
    axisFormat %s
    
    section Positive
    Precision    :0, 89
    Recall       :0, 91
    F1-Score     :0, 90
    
    section Negative
    Precision    :0, 84
    Recall       :0, 82
    F1-Score     :0, 83
    
    section Neutral
    Precision    :0, 81
    Recall       :0, 79
    F1-Score     :0, 80
    
    section Toxic
    Precision    :0, 88
    Recall       :0, 86
    F1-Score     :0, 87
```

</div>

### 📊 **Detailed Results**

```
🎭 Classification Performance:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Class     │ Precision   │   Recall    │   F1-Score  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 🟢 Positive │    0.89     │    0.91     │    0.90     │
│ 🔴 Negative │    0.84     │    0.82     │    0.83     │
│ ⚪ Neutral  │    0.81     │    0.79     │    0.80     │
│ ⚠️ Toxic    │    0.88     │    0.86     │    0.87     │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 🔮 **Demo & Usage**

<div align="center">

![Demo Animation](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=FF6B35&center=true&vCenter=true&width=700&lines=🎮+Try+Interactive+Demo;💻+Easy+Python+Integration;⚡+Real-time+Classification;🚀+Production+Ready)

### 🎮 **Interactive Demo**

[![Demo App](https://img.shields.io/badge/🎮%20Try%20Live%20Demo-Gradio%20App-ff6b35?style=for-the-badge&logo=gradio&logoColor=white)](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

</div>

<div align="center">

```mermaid
graph TD
    A[📝 Input Text] --> B{🤖 PhoBERT Classifier}
    B --> C[🟢 Positive: 85%]
    B --> D[🔴 Negative: 10%]
    B --> E[⚪ Neutral: 3%]
    B --> F[⚠️ Toxic: 2%]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#e8f5e8,stroke:#4caf50,stroke-width:3px
    style D fill:#ffebee
    style E fill:#fafafa
    style F fill:#fff3e0
    
    classDef highlight fill:#fff,stroke:#ff6b35,stroke-width:2px
```

</div>

### 💻 **Code Example**

```python
from transformers import pipeline

# 🚀 Khởi tạo pipeline
classifier = pipeline(
    "text-classification", 
    model="vanhai123/phobert-vi-comment-4class"
)

# 🔍 Phân loại bình luận
result = classifier("Tôi không đồng ý với quan điểm này")
print(f"📊 Kết quả: {result}")

# 🎯 Ví dụ nhiều câu
comments = [
    "Sản phẩm này rất tuyệt vời!",
    "Tôi không hài lòng với dịch vụ",
    "Bình thường thôi, không có gì đặc biệt",
    "Đồ rác, ai mua là ngu!"
]

for comment in comments:
    result = classifier(comment)
    print(f"💬 '{comment}' → {result[0]['label']} ({result[0]['score']:.2%})")
```

---

## 🌟 **Roadmap & Extensions**

<div align="center">

![Roadmap](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=3500&pause=1000&color=9C27B0&center=true&vCenter=true&width=600&lines=🔄+Text+Rewriting+Features;🤖+Chatbot+Integration;🛡️+Advanced+Moderation;🌍+Multi-language+Support)

### 🚀 **Planned Features**

</div>

<div align="center">

```mermaid
timeline
    title 🗓️ Development Roadmap
    
    2024 Q1 : 🔄 Text Rewriting
             : Auto suggestion
             : Tone conversion
             
    2024 Q2 : 🤖 Chatbot Integration
             : Real-time analysis
             : Smart responses
             
    2024 Q3 : 🛡️ Moderation Tools
             : Content filtering
             : Auto-moderation
             
    2024 Q4 : 🌍 Global Expansion
             : Multi-language
             : Platform APIs
```

</div>

<table>
<tr>
<td width="33%">

**🔄 Text Rewriting**
- Tự động gợi ý viết lại
- Chuyển đổi tone
- Cải thiện văn phong

</td>
<td width="33%">

**🤖 Chatbot Integration**
- Tích hợp vào chatbot
- Real-time analysis
- Smart responses

</td>
<td width="33%">

**🛡️ Moderation Tools**
- Content filtering
- Auto-moderation
- Platform integration

</td>
</tr>
</table>

### 🎯 **Future Enhancements**

- [ ] 🌐 **Multi-platform API**
- [ ] 📱 **Mobile SDK**
- [ ] 🔄 **Real-time streaming**
- [ ] 📊 **Advanced analytics**
- [ ] 🌍 **Multi-language support**
- [ ] 🧠 **Emotion detection**

---

## 🤝 **Contributing**

<div align="center">

### 💝 **Đóng góp cho dự án**

[![Contributors](https://img.shields.io/badge/Contributors-Welcome-brightgreen?style=for-the-badge&logo=github)](https://github.com/vanhai123/phobert-comment-classifier/issues)

</div>

```bash
# 🍴 Fork repository
git clone https://github.com/vanhai123/phobert-comment-classifier.git

# 🌿 Tạo branch mới
git checkout -b feature/amazing-feature

# 💾 Commit changes
git commit -m "✨ Add amazing feature"

# 🚀 Push to branch
git push origin feature/amazing-feature

# 🔄 Open Pull Request
```

---

## 📞 **Liên hệ & Hỗ trợ**

<div align="center">

![Contact Animation](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=E91E63&center=true&vCenter=true&width=500&lines=📧+Available+for+Collaboration;🤝+Open+Source+Contributor;💼+AI+%26+NLP+Specialist;🌟+Vietnamese+Tech+Community)

### 👨‍💻 **Tác giả: Hà Văn Hải**

<table>
<tr>
<td align="center">
<img src="https://github.com/vanhai123.png" width="100px" style="border-radius: 50%;" alt="Hà Văn Hải"/>
<br>
<sub><b>🇻🇳 AI Researcher</b></sub>
</td>
<td align="center">

```mermaid
mindmap
  root((👨‍💻 Contact))
    📧 Email
      vanhai11203@gmail.com
    🤗 HuggingFace
      vanhai123
    🐙 GitHub
      vanhai123
    💼 LinkedIn
      Professional Network
```

</td>
</tr>
</table>

[![Email](https://img.shields.io/badge/📧%20Email-vanhai11203@gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:vanhai11203@gmail.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-vanhai123-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/vanhai123)
[![GitHub](https://img.shields.io/badge/🐙%20GitHub-vanhai123-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vanhai123)

![Profile Views](https://komarev.com/ghpvc/?username=vanhai123&color=blueviolet&style=for-the-badge)

</div>

---

## 📄 **License & Citation**

<details>
<summary>📜 <strong>MIT License</strong></summary>

```
MIT License

Copyright (c) 2024 Hà Văn Hải

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

</details>

### 📚 **Citation**

```bibtex
@misc{phobert-vi-comment-classifier,
  title={PhoBERT Vietnamese Comment Classifier},
  author={Hà Văn Hải},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/vanhai123/phobert-vi-comment-4class}
}
```

---

<div align="center">

### 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=vanhai123/phobert-comment-classifier&type=Date)](https://star-history.com/#vanhai123/phobert-comment-classifier&Date)

![GitHub Stats](https://github-readme-stats.vercel.app/api/top-langs/?username=vanhai123&layout=compact&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=58a6ff&text_color=c9d1d9)

---

**⭐ Nếu project hữu ích, đừng quên cho một star nhé! ⭐**

<div align="center">

![Thank You](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=FFD700&center=true&vCenter=true&width=600&lines=🙏+Cảm+ơn+bạn+đã+quan+tâm!;⭐+Don't+forget+to+star+this+repo!;🚀+Happy+Coding+with+AI!;💖+Made+with+Love+in+Vietnam)

</div>

![Wave](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&animation=fadeIn)

</div>

---

> ✨ **Được phát triển với ❤️ sử dụng Hugging Face Transformers & PhoBERT trên dữ liệu tiếng Việt thực tế**
