import gradio as gr
from transformers import pipeline
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load mô hình từ Hugging Face Model Hub
pipe = pipeline("text-classification", model="vanhai123/phobert-vi-comment-4class", tokenizer="vanhai123/phobert-vi-comment-4class")

# Map nhãn lại cho dễ đọc với emoji và màu sắc
label_map = {
    "LABEL_0": {"name": "Tích cực", "emoji": "😊", "color": "#22c55e"},
    "LABEL_1": {"name": "Tiêu cực", "emoji": "😞", "color": "#ef4444"},
    "LABEL_2": {"name": "Trung tính", "emoji": "😐", "color": "#64748b"},
    "LABEL_3": {"name": "Độc hại", "emoji": "😡", "color": "#dc2626"}
}

def classify_comment(comment):
    if not comment.strip():
        return "Vui lòng nhập bình luận để phân tích!", None, None
    
    # Lấy tất cả kết quả dự đoán
    results = pipe(comment)
    if isinstance(results, list):
        results = results[0] if results else {}
    
    # Lấy kết quả chi tiết cho tất cả các nhãn
    all_scores = pipe(comment, return_all_scores=True)
    if isinstance(all_scores, list):
        all_scores = all_scores[0] if all_scores else []
    
    # Tạo kết quả chính
    main_label = results.get('label', 'UNKNOWN')
    main_score = results.get('score', 0)
    
    if main_label in label_map:
        label_info = label_map[main_label]
        main_result = f"""
        <div style="
            background: linear-gradient(135deg, {label_info['color']}22, {label_info['color']}11);
            border: 2px solid {label_info['color']};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">{label_info['emoji']}</div>
            <div style="font-size: 24px; font-weight: bold; color: {label_info['color']}; margin-bottom: 5px;">
                {label_info['name']}
            </div>
            <div style="font-size: 18px; color: #666;">
                Độ tin cậy: <strong>{round(main_score*100, 1)}%</strong>
            </div>
        </div>
        """
    else:
        main_result = f"Không xác định được nhãn: {main_label}"
    
    # Tạo biểu đồ phân phối điểm số
    if all_scores:
        labels = []
        scores = []
        colors = []
        
        for item in all_scores:
            label_key = item['label']
            if label_key in label_map:
                labels.append(label_map[label_key]['name'])
                scores.append(item['score'])
                colors.append(label_map[label_key]['color'])
        
        # Tạo biểu đồ thanh ngang
        fig = go.Figure(data=[
            go.Bar(
                y=labels,
                x=scores,
                orientation='h',
                marker_color=colors,
                text=[f"{s:.1%}" for s in scores],
                textposition='inside',
                textfont=dict(color='white', size=12, family='Arial Black')
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Phân phối điểm số dự đoán',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial', 'color': '#333'}
            },
            xaxis_title="Điểm số",
            yaxis_title="Loại bình luận",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                range=[0, 1]
            ),
            yaxis=dict(
                showgrid=False
            )
        )
        
        # Chi tiết điểm số
        details = "<div style='margin-top: 15px;'>"
        details += "<h4 style='color: #333; margin-bottom: 10px;'>📊 Chi tiết điểm số:</h4>"
        for item in sorted(all_scores, key=lambda x: x['score'], reverse=True):
            label_key = item['label']
            if label_key in label_map:
                info = label_map[label_key]
                percentage = item['score'] * 100
                bar_width = int(item['score'] * 100)
                details += f"""
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                        <span style="font-weight: 500;">{info['emoji']} {info['name']}</span>
                        <span style="font-weight: bold; color: {info['color']};">{percentage:.1f}%</span>
                    </div>
                    <div style="background: #f0f0f0; border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: {info['color']}; height: 100%; width: {bar_width}%; border-radius: 10px;"></div>
                    </div>
                </div>
                """
        details += "</div>"
        
        return main_result, fig, details
    
    return main_result, None, None

# Custom CSS cho giao diện
custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
* {
    font-family: 'Inter', sans-serif !important;
}

/* Header styling */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* Main container */
#main_container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin: 20px;
    overflow: hidden;
}

/* Title area */
.app-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
    margin: -20px -20px 30px -20px;
}

.app-title h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.app-title p {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 300;
}

/* Input styling */
.input-container textarea {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    resize: vertical !important;
}

.input-container textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

/* Button styling */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}

/* Examples styling */
.examples-container {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.examples-container h3 {
    color: #334155;
    margin-bottom: 15px;
    font-weight: 600;
}

/* Output styling */
.output-container {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-top: 20px;
}

/* Tab styling */
.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #64748b;
    font-size: 14px;
    border-top: 1px solid #e2e8f0;
    margin-top: 30px;
}
"""

# Tạo giao diện Gradio
with gr.Blocks(css=custom_css, title="Phân loại bình luận tiếng Việt - PhoBERT", theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
    <div class="app-title">
        <h1>🧠 Phân loại bình luận tiếng Việt</h1>
        <p>Sử dụng mô hình PhoBERT để phân tích cảm xúc và độc hại trong bình luận mạng xã hội</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.HTML("<h3 style='color: #334155; font-weight: 600; margin-bottom: 15px;'>📝 Nhập bình luận cần phân tích</h3>")
            
            input_text = gr.Textbox(
                lines=4,
                placeholder="Nhập bình luận tiếng Việt để phân tích cảm xúc và độ độc hại...\n\nVí dụ: 'Sản phẩm này thật tuyệt vời, tôi rất hài lòng!'",
                label="",
                elem_classes=["input-container"]
            )
            
            submit_btn = gr.Button(
                "🔍 Phân tích bình luận",
                variant="primary",
                size="lg"
            )
            
            # Examples section
            gr.HTML("""
            <div class="examples-container">
                <h3>💡 Ví dụ mẫu:</h3>
                <p style="color: #64748b; margin-bottom: 10px;">Nhấp vào các ví dụ bên dưới để thử nghiệm:</p>
            </div>
            """)
            
            gr.Examples(
                examples=[
                    "Bạn làm tốt lắm, cảm ơn nhiều! Tôi rất hài lòng với sản phẩm này.",
                    "Sản phẩm quá tệ, không đáng tiền. Chất lượng kém quá!",
                    "Tôi không có ý kiến gì đặc biệt về vấn đề này.",
                    "Mày bị điên à, nói chuyện như vậy mà cũng được?",
                    "Dịch vụ khách hàng rất tốt, nhân viên nhiệt tình hỗ trợ.",
                    "Giao hàng chậm quá, đã 1 tuần rồi mà chưa nhận được.",
                    "Thông tin này khá hữu ích, cảm ơn bạn đã chia sẻ.",
                    "Đồ rác, ai mua là ngu! Tiền bỏ ra sông bỏ ra bể."
                ],
                inputs=input_text
            )
        
        with gr.Column(scale=1):
            # Output section
            gr.HTML("<h3 style='color: #334155; font-weight: 600; margin-bottom: 15px;'>📊 Kết quả phân tích</h3>")
            
            with gr.Tabs():
                with gr.TabItem("🎯 Kết quả chính", elem_id="main_result_tab"):
                    result_output = gr.HTML(
                        value="<div style='text-align: center; padding: 40px; color: #64748b;'>Nhập bình luận và nhấn 'Phân tích' để xem kết quả</div>"
                    )
                
                with gr.TabItem("📈 Biểu đồ phân phối", elem_id="chart_tab"):
                    chart_output = gr.Plot()
                
                with gr.TabItem("📋 Chi tiết điểm số", elem_id="details_tab"):
                    details_output = gr.HTML()
    
    # Event handlers
    submit_btn.click(
        fn=classify_comment,
        inputs=input_text,
        outputs=[result_output, chart_output, details_output]
    )
    
    input_text.submit(
        fn=classify_comment,
        inputs=input_text,
        outputs=[result_output, chart_output, details_output]
    )
    
    # Footer
    gr.HTML("""
    <div class="footer">
        <p>
            <strong>Mô hình:</strong> PhoBERT fine-tuned cho phân loại bình luận tiếng Việt<br>
            <strong>Các nhãn:</strong> Tích cực • Tiêu cực • Trung tính • Độc hại<br>
            <em>Được xây dựng với ❤️ sử dụng Transformers và Gradio</em>
        </p>
    </div>
    """)

# Launch the app
demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True,
    quiet=False
)