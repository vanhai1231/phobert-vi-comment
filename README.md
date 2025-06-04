# <div align="center">📄 PhoBERT Comment Classifier (Tiếng Việt)</div>

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=6366F1&center=true&width=800&lines=Mô+hình+phân+loại+4+nhãn+cảm+xúc+trong+bình+luận+tiếng+Việt;Dựa+trên+PhoBERT+pre-trained+model;Hỗ+trợ+positive%2C+negative%2C+neutral%2C+toxic" alt="Typing SVG" />
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Accuracy-86%25-brightgreen?style=for-the-badge&logo=accuracy&logoColor=white" alt="Accuracy" />
  <img src="https://img.shields.io/badge/F1_Score-83%25-blue?style=for-the-badge&logo=f1&logoColor=white" alt="F1 Score" />
  <img src="https://img.shields.io/badge/Language-Vietnamese-red?style=for-the-badge&logo=vietnam&logoColor=white" alt="Language" />
  <img src="https://img.shields.io/badge/Model-PhoBERT-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Model" />
</div>

<br>

<div align="center">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=vanhai123&repo=phobert-vi-comment-4class&theme=radical&border_radius=10" alt="Repo Stats" />
</div>

---

## 🎯 <img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="30"> **Mục tiêu**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/😊-Positive-success?style=for-the-badge&labelColor=2d3748&color=48bb78" />
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/😞-Negative-critical?style=for-the-badge&labelColor=2d3748&color=f56565" />
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/😐-Neutral-informational?style=for-the-badge&labelColor=2d3748&color=4299e1" />
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/🤬-Toxic-important?style=for-the-badge&labelColor=2d3748&color=ed8936" />
      </td>
    </tr>
  </table>
</div>

> **Xây dựng mô hình AI dựa trên PhoBERT để phân loại bình luận tiếng Việt**
> 
> Dữ liệu được thu thập từ **TikTok**, **Facebook**, **YouTube** và các nền tảng MXH khác

---

## 📊 <img src="https://media.giphy.com/media/iY8CRBdQXODJSCERIr/giphy.gif" width="30"> **Dữ liệu**

<div align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=vanhai123&theme=react-dark&bg_color=20232a&hide_border=true" width="100%"/>
</div>

<table align="center">
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/💬-Số_bình_luận-4896-blue?style=for-the-badge&logo=comments&logoColor=white" /></td>
  </tr>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/🔗-Nguồn_dữ_liệu-TikTok_|_Facebook_|_YouTube-purple?style=for-the-badge&logo=socialmedia&logoColor=white" /></td>
  </tr>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/📋-Trường_dữ_liệu-comment_|_label_|_category-green?style=for-the-badge&logo=database&logoColor=white" /></td>
  </tr>
</table>

<div align="center">
  <a href="https://huggingface.co/datasets/vanhai123/vietnamese-social-comments">
    <img src="https://img.shields.io/badge/🤗-Dataset-Vietnamese_Social_Comments-yellow?style=for-the-badge&logo=huggingface&logoColor=white" />
  </a>
</div>

---

## 🚀 <img src="https://media.giphy.com/media/LnQjpWaON8nhr21vNW/giphy.gif" width="30"> **Cài đặt**

<div align="center">
  
```bash
# 🔧 Cài đặt các thư viện cần thiết
pip install transformers datasets scikit-learn sentencepiece

# 🎨 Tạo môi trường ảo (khuyến nghị)
python -m venv phobert_env
source phobert_env/bin/activate  # Linux/Mac
# phobert_env\Scripts\activate  # Windows
```

</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-4.21+-orange?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-1.12+-red?style=for-the-badge&logo=pytorch&logoColor=white" />
</div>

---

## 👩‍💼 <img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="30"> **Cách train mô hình**

<div align="center">

```python
# 🤖 Khởi tạo mô hình PhoBERT
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# 🔥 Bắt đầu quá trình huấn luyện
# Xem chi tiết tại train.py hoặc train.ipynb
```

</div>

<div align="center">
  <img src="https://img.shields.io/badge/📝-train.py-Available-brightgreen?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/📓-train.ipynb-Available-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
</div>

---

## 🔍 <img src="https://media.giphy.com/media/3ohhwytHcusSCXXOUg/giphy.gif" width="30"> **Kết quả**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/🎯-Accuracy-86%25-success?style=for-the-badge&labelColor=2d3748" />
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/📊-Macro_F1-83%25-informational?style=for-the-badge&labelColor=2d3748" />
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/💪-Mạnh_nhất-Positive-brightgreen?style=for-the-badge&labelColor=2d3748" />
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/🔥-Hiệu_quả-Toxic_Detection-orange?style=for-the-badge&labelColor=2d3748" />
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=vanhai123&theme=radical&border_radius=10" alt="Streak Stats" />
</div>

---

## 📱 <img src="https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif" width="30"> **Demo Gradio**

<div align="center">
  <a href="https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app">
    <img src="https://img.shields.io/badge/🚀-Live_Demo-Phobert_Comment_App-yellow?style=for-the-badge&logo=gradio&logoColor=white" />
  </a>
</div>

<div align="center">

```python
# 🎮 Sử dụng mô hình đã train
from transformers import pipeline

pipe = pipeline("text-classification", model="./phobert-4class")
result = pipe("Tôi không đồng ý với quan điểm này")

# 📊 Kết quả trả về
print(result)
# [{'label': 'negative', 'score': 0.8234}]
```

</div>

---

## 🌟 <img src="https://media.giphy.com/media/QssGEmpSoKANBbxZEb/giphy.gif" width="30"> **Gợi ý mở rộng**

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/✍️-Rewrite_Suggestion-Phân_loại_+_Gợi_ý_viết_lại-purple?style=for-the-badge&logo=edit&logoColor=white" />
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/🤖-Chatbot_Integration-Ứng_dụng_vào_chatbot_tiếng_Việt-blue?style=for-the-badge&logo=robot&logoColor=white" />
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://img.shields.io/badge/🛡️-Moderation_Tool-Dành_cho_moderation_trong_MXH-red?style=for-the-badge&logo=shield&logoColor=white" />
      </td>
    </tr>
  </table>
</div>

---

## 🗓️ <img src="https://media.giphy.com/media/mGcNjsfWAjY5AEZNw6/giphy.gif" width="30"> **Tác giả**

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=2000&pause=1000&color=F75C7E&center=true&width=400&lines=Hà+Văn+Hải;AI+Developer;NLP+Enthusiast" alt="Author" />
</div>

<div align="center">
  <a href="mailto:vanhai11203@gmail.com">
    <img src="https://img.shields.io/badge/📧-Email-vanhai11203@gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://huggingface.co/vanhai123">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face-vanhai123-yellow?style=for-the-badge&logo=huggingface&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://huggingface.co/vanhai123/phobert-vi-comment-4class">
    <img src="https://img.shields.io/badge/📁-Model-phobert_vi_comment_4class-orange?style=for-the-badge&logo=huggingface&logoColor=white" />
  </a>
</div>

---

<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=4000&pause=1000&color=36BCF7&center=true&width=800&lines=✨+Mô+hình+này+được+huấn+luyện+với+Hugging+Face+Transformers;🚀+PhoBERT+trên+dữ+liệu+tiếng+Việt+thực+tế;🎯+Đạt+độ+chính+xác+86%25+trên+4+nhãn+cảm+xúc;💡+Hỗ+trợ+phân+loại+bình+luận+MXH+Việt+Nam" alt="Footer" />
</div>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer&text=Thank%20You%20For%20Visiting!&fontSize=20&fontAlignY=65&animation=twinkling"/>
</div>

---

<div align="center">
  <img src="https://komarev.com/ghpvc/?username=vanhai123&style=for-the-badge&color=blueviolet" alt="Profile Views" />
  <img src="https://img.shields.io/github/stars/vanhai123/phobert-vi-comment-4class?style=for-the-badge&color=yellow" alt="Stars" />
  <img src="https://img.shields.io/github/forks/vanhai123/phobert-vi-comment-4class?style=for-the-badge&color=blue" alt="Forks" />
</div>
