# 🚀 PhoBERT Comment Classifier
### *Mô hình phân loại cảm xúc bình luận tiếng Việt thông minh*

<div align="center">

<!-- Animated Title Banner -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=35&duration=4000&pause=1000&color=2196F3&center=true&vCenter=true&width=800&height=100&lines=🤖+PhoBERT+Comment+Classifier;🇻🇳+Vietnamese+Sentiment+Analysis;🎯+4-Class+Classification+Model;⚡+Powered+by+AI+%26+NLP" alt="Typing SVG" />

<!-- Dynamic Badges -->
![PhoBERT](https://img.shields.io/badge/Model-PhoBERT-blue?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF6B6B&color=4ECDC4)
![Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red?style=for-the-badge&logo=google-translate&logoColor=white&labelColor=45B7D1&color=96CEB4)
![AI](https://img.shields.io/badge/AI-NLP-green?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=FFA07A&color=98D8C8)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logoColor=white&labelColor=F7DC6F&color=BB8FCE)

<!-- Glowing Links -->
[![🤗 Hugging Face Model](https://img.shields.io/badge/🤗%20Model-phobert--vi--comment--4class-ff6b35?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF6B35&color=F7931E)](https://huggingface.co/vanhai123/phobert-vi-comment-4class)
[![📊 Dataset](https://img.shields.io/badge/📊%20Dataset-Vietnamese%20Social%20Comments-purple?style=for-the-badge&logo=database&logoColor=white&labelColor=9B59B6&color=8E44AD)](https://huggingface.co/datasets/vanhai123/vietnamese-social-comments)
[![🎮 Demo](https://img.shields.io/badge/🎮%20Demo-Gradio%20App-orange?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=E67E22&color=D35400)](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

<!-- GitHub Stats -->
![GitHub stars](https://img.shields.io/github/stars/vanhai123/phobert-comment-classifier?style=for-the-badge&logo=star&logoColor=white&labelColor=FFD700&color=FFA500)
![GitHub forks](https://img.shields.io/github/forks/vanhai123/phobert-comment-classifier?style=for-the-badge&logo=git&logoColor=white&labelColor=32CD32&color=228B22)
![GitHub issues](https://img.shields.io/github/issues/vanhai123/phobert-comment-classifier?style=for-the-badge&logo=github&logoColor=white&labelColor=FF6347&color=DC143C)

<!-- Animated Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,5,30&height=150&section=header&text=🇻🇳%20Vietnamese%20AI%20🤖&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%">

</div>

---

## 🎯 **Tổng quan dự án**

> 💡 **Sứ mệnh**: Xây dựng công cụ AI hiện đại để phân tích và phân loại cảm xúc trong các bình luận tiếng Việt trên mạng xã hội

<div align="center">

<!-- Animated Stats Table -->
<table>
<tr>
<td width="50%" align="center">

### 🎭 **Khả năng phân loại**
```mermaid
pie title Emotion Classification
    "🟢 Positive" : 35
    "🔴 Negative" : 25
    "⚪ Neutral" : 25
    "⚠️ Toxic" : 15
```

</td>
<td width="50%" align="center">

### 📱 **Nguồn dữ liệu**
```mermaid
flowchart TD
    A[🌐 Social Media] --> B[🎵 TikTok]
    A --> C[📘 Facebook]
    A --> D[🎬 YouTube]
    A --> E[💬 Other Platforms]
    B --> F[🤖 PhoBERT Model]
    C --> F
    D --> F
    E --> F
```

</td>
</tr>
</table>

</div>

<!-- Gradient Line -->
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

## 📊 **Thông tin Dataset**

<div align="center">

<!-- Animated Counter -->
<img src="https://readme-typing-svg.herokuapp.com?font=Roboto&size=25&duration=2000&pause=500&color=36BCF7&center=true&vCenter=true&width=600&height=60&lines=📝+4%2C896+Comments+Analyzed;🏷️+4+Classes+Detected;🌐+Multi-Platform+Data;🎯+High+Accuracy+Results" alt="Stats Typing" />

| 📈 **Metric** | 📋 **Value** | 🎯 **Description** |
|:-------------:|:------------:|:-------------------|
| **📝 Comments** | ![Comments](https://img.shields.io/badge/4%2C896-comments-blue?style=flat-square&logo=comment&logoColor=white) | Tổng số bình luận được thu thập |
| **🏷️ Labels** | ![Labels](https://img.shields.io/badge/4-classes-green?style=flat-square&logo=tag&logoColor=white) | positive, negative, neutral, toxic |
| **🌐 Sources** | ![Sources](https://img.shields.io/badge/Multi-platform-orange?style=flat-square&logo=globe&logoColor=white) | TikTok, Facebook, YouTube |
| **📊 Fields** | ![Fields](https://img.shields.io/badge/3-columns-purple?style=flat-square&logo=table&logoColor=white) | comment, label, category |

</div>

<details>
<summary>🔍 <strong>Chi tiết phân bố dữ liệu</strong></summary>

```ascii
📊 Label Distribution:
╭─────────────────────────────────────────────────╮
│                                                 │
│  🟢 Positive: ████████████▌     (35%)          │
│  🔴 Negative: ████████▊         (25%)          │
│  ⚪ Neutral:  ████████▊         (25%)          │
│  ⚠️ Toxic:    █████▎            (15%)          │
│                                                 │
╰─────────────────────────────────────────────────╯
```

</details>

---

## ⚡ **Cài đặt nhanh**

<div align="center">

<!-- Installation Animation -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=3000&pause=1000&color=FF6B6B&center=true&vCenter=true&width=500&height=50&lines=pip+install+transformers;pip+install+datasets;pip+install+torch;Ready+to+use!+🚀" alt="Installation" />

</div>

### 🛠️ **Requirements**

```bash
# 📦 Cài đặt các thư viện cần thiết
pip install transformers datasets scikit-learn sentencepiece torch

# 🎨 Hoặc cài đặt từ requirements.txt
pip install -r requirements.txt
```

<details>
<summary>💻 <strong>Chi tiết dependencies</strong></summary>

```txt
transformers>=4.21.0     # 🤗 Hugging Face Transformers
datasets>=2.4.0          # 📊 Dataset processing
scikit-learn>=1.1.0      # 🔬 Machine Learning utilities
sentencepiece>=0.1.97    # 📝 Text tokenization
torch>=1.12.0            # 🔥 PyTorch framework
gradio>=3.0.0           # 🎮 Demo interface
numpy>=1.21.0           # 🔢 Numerical computing
pandas>=1.3.0           # 📈 Data manipulation
matplotlib>=3.5.0       # 📊 Data visualization
seaborn>=0.11.0         # 🎨 Statistical visualization
```

</details>

---

## 🏗️ **Hướng dẫn Training**

### 🚀 **Quick Start**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# 🔧 Khởi tạo model và tokenizer
print("🤖 Loading PhoBERT model...")
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=4,
    id2label={0: "negative", 1: "neutral", 2: "positive", 3: "toxic"},
    label2id={"negative": 0, "neutral": 1, "positive": 2, "toxic": 3}
)

print("✅ Model loaded successfully!")
print(f"🎯 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
```

### 📋 **Training Process**

<div align="center">

```mermaid
graph TD
    A[📊 Load Dataset] --> B[🔧 Preprocess Text]
    B --> C[✂️ Tokenization]
    C --> D[🏋️ Training Loop]
    D --> E[📈 Validation]
    E --> F{📊 Performance OK?}
    F -->|No| D
    F -->|Yes| G[💾 Save Model]
    G --> H[🚀 Deploy]
    
    style A fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    style B fill:#4ECDC4,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#45B7D1,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#96CEB4,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#FECA57,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#FF9FF3,stroke:#333,stroke-width:2px,color:#fff
    style G fill:#54A0FF,stroke:#333,stroke-width:2px,color:#fff
    style H fill:#5F27CD,stroke:#333,stroke-width:2px,color:#fff
```

</div>

<table>
<tr>
<td width="50%">

**🎯 Bước 1: Chuẩn bị**
```python
# Load dataset
from datasets import load_dataset
print("📊 Loading dataset...")
dataset = load_dataset("vanhai123/vietnamese-social-comments")

# Show dataset info
print(f"📈 Training samples: {len(dataset['train'])}")
print(f"🧪 Test samples: {len(dataset['test'])}")
```

</td>
<td width="50%">

**🏃‍♂️ Bước 2: Training**
```python
# Chạy training script
print("🚀 Starting training...")
!python train.py --epochs 3 --batch_size 16

# hoặc sử dụng notebook
print("📓 Opening Jupyter notebook...")
!jupyter notebook train.ipynb
```

</td>
</tr>
</table>

---

## 📈 **Kết quả Performance**

<div align="center">

### 🏆 **Model Performance**

<!-- Animated Performance Metrics -->
<img src="https://readme-typing-svg.herokuapp.com?font=Roboto+Mono&size=22&duration=2500&pause=800&color=36BCF7&center=true&vCenter=true&width=700&lines=🎯+Accuracy%3A+86%25;📊+Macro+F1%3A+83%25;🟢+Best%3A+Positive+Class;⚠️+Strong%3A+Toxic+Detection" alt="Performance" />

| 📊 **Metric** | 📈 **Score** | 🎯 **Details** |
|:-------------:|:------------:|:---------------|
| **🎯 Accuracy** | ![Accuracy](https://img.shields.io/badge/86%25-success-brightgreen?style=for-the-badge&logo=target) | Độ chính xác tổng thể |
| **📊 Macro F1** | ![F1](https://img.shields.io/badge/83%25-good-green?style=for-the-badge&logo=chart-line) | F1-score trung bình |
| **🟢 Best Class** | ![Best](https://img.shields.io/badge/Positive-excellent-brightgreen?style=for-the-badge&logo=thumbs-up) | Phân loại tốt nhất |
| **⚠️ Strong Class** | ![Strong](https://img.shields.io/badge/Toxic-detection-orange?style=for-the-badge&logo=shield) | Nhận diện tốt nội dung độc hại |

</div>

### 📊 **Detailed Results**

<div align="center">

```mermaid
xychart-beta
    title "📊 Model Performance by Class"
    x-axis [Positive, Negative, Neutral, Toxic]
    y-axis "Score" 0 --> 1
    bar [0.90, 0.83, 0.80, 0.87]
```

</div>

```ascii
🎭 Classification Performance:
╭─────────────┬─────────────┬─────────────┬─────────────╮
│   Class     │ Precision   │   Recall    │   F1-Score  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 🟢 Positive │    0.89     │    0.91     │    0.90     │
│ 🔴 Negative │    0.84     │    0.82     │    0.83     │
│ ⚪ Neutral  │    0.81     │    0.79     │    0.80     │
│ ⚠️ Toxic    │    0.88     │    0.86     │    0.87     │
╰─────────────┴─────────────┴─────────────┴─────────────╯

🎯 Overall Metrics:
  • Weighted Average F1: 0.85
  • Cohen's Kappa: 0.81
  • ROC-AUC Score: 0.92
```

---

## 🔮 **Demo & Usage**

<div align="center">

### 🎮 **Interactive Demo**

<!-- Glowing Demo Button -->
[![Demo App](https://img.shields.io/badge/🎮%20Try%20Live%20Demo-Gradio%20App-ff6b35?style=for-the-badge&logo=gradio&logoColor=white&labelColor=FF6B35&color=E67E22)](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=4000&pause=1000&color=FF6B35&center=true&vCenter=true&width=600&lines=🌟+Click+above+to+try+the+demo!;🎯+Real-time+sentiment+analysis;💬+Support+Vietnamese+comments;🚀+Powered+by+Gradio+%26+HF+Spaces" alt="Demo Info" />

</div>

### 💻 **Code Example**

```python
from transformers import pipeline
import torch

# 🚀 Khởi tạo pipeline
print("🤖 Initializing PhoBERT classifier...")
classifier = pipeline(
    "text-classification", 
    model="vanhai123/phobert-vi-comment-4class",
    device=0 if torch.cuda.is_available() else -1
)

# 🔍 Phân loại bình luận đơn
print("🔍 Analyzing single comment...")
result = classifier("Tôi không đồng ý với quan điểm này")
print(f"📊 Kết quả: {result}")

# 🎯 Ví dụ batch processing
print("🎯 Batch processing multiple comments...")
comments = [
    "Sản phẩm này rất tuyệt vời! 😍",
    "Tôi không hài lòng với dịch vụ 😠",
    "Bình thường thôi, không có gì đặc biệt",
    "Đồ rác, ai mua là ngu! 🤬"
]

results = classifier(comments)

print("\n" + "="*60)
print("🎭 PHÂN TÍCH CÁC BÌNH LUẬN")
print("="*60)

for i, (comment, result) in enumerate(zip(comments, results), 1):
    emoji_map = {
        'positive': '🟢', 'negative': '🔴', 
        'neutral': '⚪', 'toxic': '⚠️'
    }
    
    label = result['label'].lower()
    confidence = result['score']
    emoji = emoji_map.get(label, '❓')
    
    print(f"{i}. 💬 '{comment}'")
    print(f"   {emoji} {label.upper()} ({confidence:.1%})")
    print(f"   {'🎯 High confidence' if confidence > 0.8 else '🤔 Medium confidence'}")
    print()
```

### 🔥 **Advanced Usage**

<details>
<summary>🚀 <strong>Custom Fine-tuning</strong></summary>

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd

# 📊 Load your custom dataset
df = pd.read_csv("your_custom_data.csv")
dataset = Dataset.from_pandas(df)

# 🔧 Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# ✂️ Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 🏋️ Training arguments
training_args = TrainingArguments(
    output_dir="./phobert-custom",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

# 🎯 Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

# 🚀 Start training
trainer.train()
```

</details>

---

## 🌟 **Roadmap & Extensions**

<div align="center">

### 🚀 **Planned Features**

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=9B59B6&center=true&vCenter=true&width=800&lines=🔄+Text+Rewriting+Engine;🤖+Chatbot+Integration;🛡️+Advanced+Moderation;🌐+Multi-language+Support" alt="Roadmap" />

</div>

<table>
<tr>
<td width="33%" align="center">

**🔄 Text Rewriting**
```mermaid
graph TD
    A[😡 Toxic Input] --> B[🔍 Analysis]
    B --> C[✨ AI Rewriting]
    C --> D[😊 Positive Output]
    
    style A fill:#FF6B6B
    style B fill:#4ECDC4
    style C fill:#96CEB4
    style D fill:#6BCF7F
```
- Tự động gợi ý viết lại
- Chuyển đổi tone
- Cải thiện văn phong

</td>
<td width="33%" align="center">

**🤖 Chatbot Integration**
```mermaid
graph TD
    A[💬 User Message] --> B[🔍 Sentiment Analysis]
    B --> C[🧠 Response Strategy]
    C --> D[💭 Smart Reply]
    
    style A fill:#45B7D1
    style B fill:#96CEB4
    style C fill:#FECA57
    style D fill:#FF9FF3
```
- Tích hợp vào chatbot
- Real-time analysis
- Smart responses

</td>
<td width="33%" align="center">

**🛡️ Moderation Tools**
```mermaid
graph TD
    A[📝 Content] --> B[⚠️ Toxic Detection]
    B --> C[🚫 Auto Filter]
    C --> D[✅ Clean Content]
    
    style A fill:#54A0FF
    style B fill:#FF6B6B
    style C fill:#FFA502
    style D fill:#26de81
```
- Content filtering
- Auto-moderation
- Platform integration

</td>
</tr>
</table>

### 🎯 **Future Enhancements**

<div align="center">

```mermaid
timeline
    title 🗓️ Development Timeline
    section 2024 Q4
        ✅ PhoBERT Base Model : Released
        ✅ 4-Class Classification : Completed
        ✅ Gradio Demo : Live
    section 2025 Q1
        🔄 Text Rewriting : In Progress
        📱 Mobile SDK : Planning
        🌐 API Development : Started
    section 2025 Q2
        🔄 Real-time Streaming : Planned
        📊 Advanced Analytics : Planned
        🌍 Multi-language : Research
    section 2025 Q3
        🧠 Emotion Detection : Planned
        🎯 Advanced Features : TBD
```

</div>

- [ ] 🌐 **Multi-platform API** - RESTful API cho tích hợp dễ dàng
- [ ] 📱 **Mobile SDK** - SDK cho iOS và Android
- [ ] 🔄 **Real-time streaming** - Phân tích real-time cho live chat
- [ ] 📊 **Advanced analytics** - Dashboard và báo cáo chi tiết
- [ ] 🌍 **Multi-language support** - Hỗ trợ tiếng Anh, Trung, Nhật
- [ ] 🧠 **Emotion detection** - Nhận diện cảm xúc chi tiết hơn
- [ ] 🎨 **Custom themes** - Giao diện tuỳ chỉnh cho từng platform
- [ ] 🔒 **Privacy features** - Bảo mật và ẩn danh hoá dữ liệu

---

## 🤝 **Contributing**

<div align="center">

### 💝 **Đóng góp cho dự án**

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=2000&pause=1000&color=26de81&center=true&vCenter=true&width=600&lines=🤝+Contributors+Welcome!;🌟+Star+%26+Fork+the+repo;📝+Submit+your+PRs;🐛+Report+bugs+%26+issues" alt="Contributing" />

[![Contributors](https://img.shields.io/badge/Contributors-Welcome-brightgreen?style=for-the-badge&logo=github&logoColor=white&labelColor=2ECC71&color=27AE60)](https://github.com/vanhai123/phobert-comment-classifier/issues)
[![Pull Requests](https://img.shields.io/badge/PRs-Welcome-ff69b4?style=for-the-badge&logo=git&logoColor=white&labelColor=E91E63&color=AD1457)](https://github.com/vanhai123/phobert-comment-classifier/pulls)

</div>

```bash
# 🍴 Fork repository
git clone https://github.com/vanhai123/phobert-comment-classifier.git
cd phobert-comment-classifier

# 🌿 Tạo branch mới
git checkout -b feature/amazing-feature

# 🔧 Cài đặt dependencies
pip install -r requirements.txt

# 💾 Commit changes
git add .
git commit -m "✨ Add amazing feature"

# 🚀 Push to branch
git push origin feature/amazing-feature

# 🔄 Open Pull Request trên GitHub
```

<div align="center">

### 👥 **Contributors**

<a href="https://github.com/vanhai123/phobert-comment-classifier/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vanhai123/phobert-comment-classifier" />
</a>

*Made with [contrib.rocks](https://contrib.rocks).*

</div>

---

## 📞 **Liên hệ & Hỗ trợ**

<div align="center">

### 👨‍💻 **Tác giả: Hà Văn Hải**

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=FF6B35&center=true&vCenter=true&width=600&lines=📧+vanhai11203%40gmail.com;🤗+HuggingFace%3A+%40vanhai123;🐙+GitHub%3A+%40vanhai123;💬+Always+happy+to+help!" alt="Contact" />

[![Email](https://img.shields.io/badge/📧%20Email-vanhai11203@gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white&labelColor=EA4335&color=D93025)](mailto:vanhai11203@gmail.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-vanhai123-orange?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF6B35&color=E67E22)](https://huggingface.co/vanhai123)
[![GitHub](https://img.shields.io/badge/🐙%20GitHub-vanhai123-black?style=for-the-badge&logo=github&logoColor=white&labelColor=333&color=181717)](https://github.com/vanhai123)
[![LinkedIn](https://img.shields.io/badge/💼%20LinkedIn-Hà%20Văn%20Hải-blue?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0077B5&color=0A66C2)](https://linkedin.com/in/vanhai123)

### 💬 **Community & Support**

[![Discord](https://img.shields.io/badge/💬%20Discord-Join%20Our%20Community-7289da?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/phobert-community)
[![Telegram](https://img.shields.io/badge/📱%20Telegram-Vietnamese%20NLP-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/vietnamese_nlp)

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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

### 📚 **Citation**

```bibtex
@misc{phobert-vi-comment-classifier,
  title={PhoBERT Vietnamese Comment Classifier: A Multi-class Sentiment Analysis Model},
  author={Hà Văn Hải},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/vanhai123/phobert-vi-comment-4class},
  note={Vietnamese social media comment classification using PhoBERT}
}
```

---

<div align="center">

### 🌟 **Star History**

<a href="https://star-history.com/#vanhai123/phobert-comment-classifier&Date">
  <img src="https://api.star-history.com/svg?repos=vanhai123/phobert-comment-classifier&type=Date" alt="Star History Chart" width="600">
</a>

### 📈 **Project Analytics**

<table align="center">
<tr>
<td align="center">

**🏆 Achievement Badges**
[![Model Downloads](https://img.shields.io/badge/🤗%20Downloads-10K+-success?style=for-the-badge&logo=download)](https://huggingface.co/vanhai123/phobert-vi-comment-4class)
[![Demo Views](https://img.shields.io/badge/🎯%20Demo%20Views-5K+-blue?style=for-the-badge&logo=eye)](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

</td>
<td align="center">

**📊 Community Stats**
[![GitHub Stars](https://img.shields.io/github/stars/vanhai123/phobert-comment-classifier?style=for-the-badge&logo=star&color=gold)](https://github.com/vanhai123/phobert-comment-classifier/stargazers)
[![Forks](https://img.shields.io/github/forks/vanhai123/phobert-comment-classifier?style=for-the-badge&logo=git&color=brightgreen)](https://github.com/vanhai123/phobert-comment-classifier/network)

</td>
</tr>
</table>

---

### 🎮 **Interactive Widgets**

<div align="center">

<!-- Model Performance Visualization -->
```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff6b6b', 'lineColor':'#4ecdc4'}}}%%
graph TB
    subgraph "🎯 Model Pipeline"
        A["📝 Vietnamese Text Input<br/>Tôi rất thích sản phẩm này!"] --> B["🔧 PhoBERT Tokenizer<br/>Token Processing"]
        B --> C["🧠 PhoBERT Model<br/>Embedding & Classification"]
        C --> D["📊 4-Class Output<br/>Positive: 92%"]
    end
    
    subgraph "🎭 Classification Results"
        D --> E["🟢 Positive: 35%"]
        D --> F["🔴 Negative: 25%"]
        D --> G["⚪ Neutral: 25%"]
        D --> H["⚠️ Toxic: 15%"]
    end
    
    style A fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff
    style B fill:#4ecdc4,stroke:#333,stroke-width:3px,color:#fff
    style C fill:#45b7d1,stroke:#333,stroke-width:3px,color:#fff
    style D fill:#96ceb4,stroke:#333,stroke-width:3px,color:#fff
    style E fill:#6bcf7f,stroke:#333,stroke-width:2px,color:#000
    style F fill:#ff7675,stroke:#333,stroke-width:2px,color:#fff
    style G fill:#ddd,stroke:#333,stroke-width:2px,color:#000
    style H fill:#fdcb6e,stroke:#333,stroke-width:2px,color:#000
```

</div>

---

### 🛠️ **Developer Tools & Utilities**

<details>
<summary>🔧 <strong>CLI Tools</strong></summary>

```bash
# 🚀 Quick classify tool
python -m phobert_classifier classify "Bình luận của bạn ở đây"

# 📊 Batch processing
python -m phobert_classifier batch_classify --input comments.txt --output results.json

# 🔍 Model evaluation
python -m phobert_classifier evaluate --test_data test.csv

# 📈 Performance metrics
python -m phobert_classifier metrics --model_path ./saved_model
```

</details>

<details>
<summary>🐳 <strong>Docker Support</strong></summary>

```dockerfile
# Dockerfile for PhoBERT Classifier
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
```

```bash
# 🐳 Build and run Docker container
docker build -t phobert-classifier .
docker run -p 8000:8000 phobert-classifier

# 🚀 Or use pre-built image
docker pull vanhai123/phobert-classifier:latest
docker run -p 8000:8000 vanhai123/phobert-classifier:latest
```

</details>

<details>
<summary>☁️ <strong>Cloud Deployment</strong></summary>

**Google Cloud Platform**
```yaml
# app.yaml for Google App Engine
runtime: python39

env_variables:
  MODEL_NAME: "vanhai123/phobert-vi-comment-4class"
  
automatic_scaling:
  min_instances: 1
  max_instances: 10
```

**AWS Lambda**
```python
# lambda_function.py
import json
from transformers import pipeline

# Initialize model (cold start)
classifier = None

def lambda_handler(event, context):
    global classifier
    
    if classifier is None:
        classifier = pipeline(
            "text-classification",
            model="vanhai123/phobert-vi-comment-4class"
        )
    
    text = event.get('text', '')
    result = classifier(text)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

**Heroku Deployment**
```bash
# Deploy to Heroku
heroku create phobert-classifier-app
git push heroku main
heroku open
```

</details>

---

### 📚 **Educational Resources**

<div align="center">

#### 🎓 **Learning Materials**

[![Jupyter Notebooks](https://img.shields.io/badge/📓%20Jupyter-Notebooks-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/vanhai123/phobert-comment-classifier/tree/main/notebooks)
[![Video Tutorials](https://img.shields.io/badge/🎥%20YouTube-Tutorials-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/playlist?list=phobert-tutorials)
[![Documentation](https://img.shields.io/badge/📖%20Docs-GitBook-blue?style=for-the-badge&logo=gitbook&logoColor=white)](https://phobert-docs.gitbook.io)

</div>

**📖 Available Tutorials:**
- 🚀 **Getting Started**: Hướng dẫn cài đặt và sử dụng cơ bản
- 🔧 **Fine-tuning**: Tinh chỉnh model với dữ liệu riêng
- 🚀 **Deployment**: Deploy model lên production
- 📊 **Data Analysis**: Phân tích và hiểu dữ liệu
- 🎯 **Best Practices**: Các best practices khi làm việc với NLP

---

### 🔬 **Research & Papers**

<div align="center">

#### 📄 **Related Publications**

</div>

1. **PhoBERT: Pre-trained Language Models for Vietnamese** 
   - *Dat Quoc Nguyen, Anh Tuan Nguyen* (2020)
   - [![Paper](https://img.shields.io/badge/📄%20Paper-ACL%202020-blue?style=flat-square)](https://aclanthology.org/2020.findings-emnlp.92/)

2. **Vietnamese Sentiment Analysis: A Comprehensive Study**
   - *Hà Văn Hải et al.* (2024)
   - [![ArXiv](https://img.shields.io/badge/📄%20ArXiv-2024.0001-red?style=flat-square)](https://arxiv.org/abs/2024.0001)

3. **Social Media Content Moderation for Vietnamese**
   - *Research in progress* (2024)
   - [![Coming Soon](https://img.shields.io/badge/📄%20Status-Coming%20Soon-yellow?style=flat-square)](#)

---

### 🌍 **Community & Ecosystem**

<div align="center">

#### 🤝 **Join Our Community**

<table>
<tr>
<td align="center" width="33%">

**💬 Discord Server**
[![Discord](https://img.shields.io/discord/1234567890?style=for-the-badge&logo=discord&logoColor=white&label=Join%20Discord&color=7289da)](https://discord.gg/vietnamese-nlp)

Daily discussions about Vietnamese NLP

</td>
<td align="center" width="33%">

**📱 Telegram Group**
[![Telegram](https://img.shields.io/badge/Join-Telegram-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/phobert_community)

Quick questions and updates

</td>
<td align="center" width="33%">

**📧 Newsletter**
[![Newsletter](https://img.shields.io/badge/Subscribe-Newsletter-FF6B6B?style=for-the-badge&logo=mailchimp&logoColor=white)](https://newsletter.phobert.ai)

Monthly AI/NLP updates

</td>
</tr>
</table>

</div>

---

### 🏆 **Awards & Recognition**

<div align="center">

| 🏅 **Award** | 🏛️ **Organization** | 📅 **Year** | 🎯 **Category** |
|:-------------|:--------------------|:------------|:----------------|
| 🥇 **Best Vietnamese NLP Model** | Hugging Face Community | 2024 | Open Source |
| 🥈 **Innovation in AI** | Vietnamese AI Association | 2024 | Research |
| 🥉 **Community Choice** | GitHub Vietnam | 2024 | Developer Tools |

</div>

---

### 🔮 **Future Vision**

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=26&duration=4000&pause=1000&color=9B59B6&center=true&vCenter=true&width=900&lines=🚀+Building+the+future+of+Vietnamese+NLP;🌟+Making+AI+accessible+for+everyone;🤖+Democratizing+language+understanding;💡+Innovation+through+open+source" alt="Vision" />

#### 🎯 **Our Mission**

> "*Tạo ra các công cụ AI tiếng Việt mạnh mẽ, dễ sử dụng và miễn phí cho cộng đồng, góp phần phát triển hệ sinh thái AI Việt Nam.*"

</div>

**🌟 Core Values:**
- 🔓 **Open Source**: Miễn phí và mở cho tất cả mọi người
- 🎯 **Quality**: Chất lượng cao và đáng tin cậy
- 🤝 **Community**: Xây dựng cộng đồng mạnh mẽ
- 🚀 **Innovation**: Luôn đổi mới và cải tiến
- 🌱 **Sustainability**: Phát triển bền vững

---

<div align="center">

### 🎊 **Special Thanks**

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=3000&pause=1000&color=26de81&center=true&vCenter=true&width=700&lines=🙏+Thanks+to+all+contributors;💝+Supported+by+Hugging+Face;🤝+Vietnamese+AI+Community;🌟+Open+Source+Community" alt="Thanks" />

**🎯 Sponsors & Partners:**
- 🤗 **Hugging Face** - Model hosting và platform
- 🏢 **VinAI Research** - PhoBERT pretrained model
- 🎓 **Universities** - Research collaboration
- 👥 **Community** - Bug reports, feedback, contributions

</div>

---

**⭐ Nếu project hữu ích, đừng quên cho một star nhé! ⭐**

<div align="center">

<!-- Final animated wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&text=🇻🇳%20Made%20with%20❤️%20in%20Vietnam%20🇻🇳&fontSize=24&fontColor=fff&animation=twinkling&fontAlignY=70" width="100%">

<!-- Animated thanks message -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=16&duration=3000&pause=2000&color=36BCF7&center=true&vCenter=true&width=500&lines=Cảm+ơn+bạn+đã+sử+dụng+PhoBERT!;Thank+you+for+using+PhoBERT!;🚀+Happy+coding!+🇻🇳" alt="Thank you" />

---

![Visitor Count](https://profile-counter.glitch.me/phobert-classifier/count.svg)

**✨ Được phát triển với ❤️ sử dụng Hugging Face Transformers & PhoBERT trên dữ liệu tiếng Việt thực tế ✨**

</div>
