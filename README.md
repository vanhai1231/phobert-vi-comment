




🚀 PhoBERT Comment Classifier
Mô hình phân loại cảm xúc bình luận tiếng Việt thông minh













🎯 Tổng quan dự án

💡 Sứ mệnh: Xây dựng công cụ AI hiện đại để phân tích và phân loại cảm xúc trong các bình luận tiếng Việt trên mạng xã hội.





🎭 Khả năng phân loại


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


🟢 Positive - Tích cực
🔴 Negative - Tiêu cực  
⚪ Neutral - Trung lập
⚠️ Toxic - Kích động, phản cảm





📱 Nguồn dữ liệu


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


🎵 TikTok Comments
📘 Facebook Posts
🎬 YouTube Reviews
🌐 Các platform khác







📊 Thông tin Dataset





📈 Metric
📋 Value
🎯 Description



📝 Comments
4,896
Tổng số bình luận được thu thập


🏷️ Labels
4 classes
positive, negative, neutral, toxic


🌐 Sources
Multi-platform
TikTok, Facebook, YouTube


📊 Fields
3 columns
comment, label, category


pie title Dataset Distribution
    "🟢 Positive" : 35
    "🔴 Negative" : 25
    "⚪ Neutral" : 25
    "⚠️ Toxic" : 15




🔍 Chi tiết phân bố dữ liệu

📊 Label Distribution:
├── 🟢 Positive: ~35%
├── 🔴 Negative: ~25% 
├── ⚪ Neutral:  ~25%
└── ⚠️ Toxic:    ~15%




⚡ Cài đặt nhanh
🛠️ Requirements
pip install transformers datasets scikit-learn sentencepiece torch gradio


💻 Chi tiết dependencies

transformers>=4.21.0
datasets>=2.4.0
scikit-learn>=1.1.0
sentencepiece>=0.1.97
torch>=1.12.0
gradio>=3.0.0  # Cho demo app




🏗️ Hướng dẫn Training





🚀 Quick Start
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔧 Khởi tạo model và tokenizer
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=4
)

📋 Training Process




🎯 Bước 1: Chuẩn bị
# Load dataset
from datasets import load_dataset
dataset = load_dataset("vanhai123/vietnamese-social-comments")




🏃‍♂️ Bước 2: Training
# Chạy training script
python train.py
# hoặc sử dụng notebook
jupyter notebook train.ipynb






📈 Kết quả Performance



🏆 Model Performance



📊 Metric
📈 Score
🎯 Details



🎯 Accuracy
~86%
Độ chính xác tổng thể


📊 Macro F1
~83%
F1-score trung bình


🟢 Best Class
Positive
Phân loại tốt nhất


⚠️ Strong Class
Toxic
Nhận diện tốt nội dung độc hại




📊 Detailed Results
🎭 Classification Performance:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Class     │ Precision   │   Recall    │   F1-Score  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 🟢 Positive │    0.89     │    0.91     │    0.90     │
│ 🔴 Negative │    0.84     │    0.82     │    0.83     │
│ ⚪ Neutral  │    0.81     │    0.79     │    0.80     │
│ ⚠️ Toxic    │    0.88     │    0.86     │    0.87     │
└─────────────┴─────────────┴─────────────┴─────────────┘


🔮 Demo & Usage



🎮 Interactive Demo



💻 Code Example
from transformers import pipeline

# 🚀 Khởi tạo pipeline
classifier = pipeline(
    "text-classification", 
    model="vanhai123/phobert-vi-comment-4class"
)

# 🔍 Phân loại bình luận
comments = [
    "Sản phẩm này rất tuyệt vời!",
    "Tôi không hài lòng với dịch vụ",
    "Bình thường thôi, không có gì đặc biệt",
    "Đồ rác, ai mua là ngu!"
]

for comment in comments:
    result = classifier(comment)
    print(f"💬 '{comment}' → {result[0]['label']} ({result[0]['score']:.2%})")


🌟 Roadmap & Extensions







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



🎯 Future Enhancements

 🌐 Multi-platform API
 📱 Mobile SDK
 🔄 Real-time streaming
 📊 Advanced analytics
 🌍 Multi-language support
 🧠 Emotion detection


🤝 Contributing
# 🍴 Fork repository
git clone https://github.com/vanhai123/phobert-comment-classifier.git

# 🌿 Tạo branch mới
git checkout -b feature/amazing-feature

# 💾 Commit changes
git commit -m "✨ Add amazing feature"

# 🚀 Push to branch
git push origin feature/amazing-feature

# 🔄 Open Pull Request


📞 Liên hệ & Hỗ trợ


👨‍💻 Tác giả: Hà Văn Hải





🇻🇳 AI Researcher



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









📄 License & Citation

📜 MIT License

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



📚 Citation
@misc{phobert-vi-comment-classifier,
  title={PhoBERT Vietnamese Comment Classifier},
  author={Hà Văn Hải},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/vanhai123/phobert-vi-comment-4class}
}




⭐ Nếu project hữu ích, đừng quên cho một star nhé! ⭐


