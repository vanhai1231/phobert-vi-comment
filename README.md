# 📄 Phobert Comment Classifier (Tiếng Việt)

> Mô hình phân loại 4 nhãn cảm xúc trong bình luận tiếng Việt dựa trên PhoBERT

---

## 📈 Mục tiêu

Xây dựng mô hình AI dựa trên mô hình pre-trained PhoBERT, nhằm phân loại các bình luận tiếng Việt theo 4 nhãn:

* `positive` (tích cực)
* `negative` (tiêu cực)
* `neutral` (trung lập)
* `toxic` (kích động, phản cảm)

Dữ liệu thu thập từ TikTok, Facebook, YouTube...

---

## 📊 Dữ liệu

* Số bình luận: 4.896
* Thu thập từ: TikTok, Facebook, YouTube, v.v.
* Gồm 3 trường: `comment`, `label`, `category`
* Nhãn phân loại: `positive`, `negative`, `neutral`, `toxic`
* 📅 Dataset: [Vietnamese Social Comments](https://huggingface.co/datasets/vanhai123/vietnamese-social-comments)

---

## 🚀 Cài đặt

```bash
pip install transformers datasets scikit-learn sentencepiece
```

---

## 👩‍💼 Cách train mô hình

```python
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
```

Toàn bộ quy trình train đã được viết trong `train.py` hoặc [notebook](./train.ipynb)

---

## 🔍 Kết quả

* Accuracy ≃86%
* Macro F1 ≃83%
* Mạnh nhất với lớp `positive` và `toxic`

---

## 📱 Demo Gradio (tùy chọn)

🔗 App demo: [Phobert Comment App](https://huggingface.co/spaces/vanhai123/phobert-vi-comment-app)

```python
from transformers import pipeline
pipe = pipeline("text-classification", model="./phobert-4class")
pipe("Tôi không đồng ý với quan điểm này")
```

---

## 🌟 Gợi ý mở rộng

* Kết hợp phân loại + gợi ý viết lại câu (rewrite)
* Ứng dụng vào chatbot tiếng Việt
* Dành cho moderation tool trong MXH

---

## 🗓️ Tác giả

**Hà Văn Hải**
Email: [vanhai11203@gmail.com](mailto:vanhai11203@gmail.com)
HF: [vanhai123](https://huggingface.co/vanhai123)
📁 Model: [phobert-vi-comment-4class](https://huggingface.co/vanhai123/phobert-vi-comment-4class)

---

> ✨ Mô hình này được huấn luyện với Hugging Face Transformers & PhoBERT trên dữ liệu tiếng Việt thực tế.
