import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BENCHMARK_CACHE_PATH = PROCESSED_DIR / "benchmark_results.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="Hệ thống Nhận diện Ý định NPC",
    layout="wide"
)

INTENT_NAMES = ["GREETING", "TRADE", "INFO", "HOSTILE", "UNKNOWN"]


def fetch_available_models(api_url: str):
    """Gọi GET /models để lấy danh sách mô hình từ FastAPI server"""
    try:
        resp = requests.get(f"{api_url}/models", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", []), None
    except requests.exceptions.ConnectionError:
        return [], "CONNECTION_ERROR"
    except requests.exceptions.Timeout:
        return [], "TIMEOUT_ERROR"
    except Exception as e:
        return [], str(e)


def call_generate_api(api_url: str, model_type: str, prompt: str):
    """Gọi POST /generate gửi câu thoại tới FastAPI server và đo độ trễ"""
    payload = {
        "model_type": model_type,
        "prompt": prompt
    }
    start = time.perf_counter()
    try:
        resp = requests.post(f"{api_url}/generate", json=payload, timeout=10)
        latency_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return data.get("result", "UNKNOWN"), latency_ms, None
        return None, latency_ms, f"Lỗi API ({resp.status_code}): {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, 0.0, "CONNECTION_ERROR"
    except requests.exceptions.Timeout:
        return None, 0.0, "TIMEOUT_ERROR"
    except Exception as e:
        return None, 0.0, str(e)


@st.cache_data
def load_benchmark_data():
    """Đọc kết quả kiểm thử từ file cache JSON đã chạy trước đó"""
    if not BENCHMARK_CACHE_PATH.exists():
        from backend.benchmark import run_benchmark
        return run_benchmark(save_cache=True)
    with open(BENCHMARK_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# SIDEBAR CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.header("Cài đặt hệ thống")

    api_base_url = st.text_input(
        label="API Endpoint URL",
        value="http://127.0.0.1:8000",
        help="Nhập địa chỉ FastAPI Backend (Local hoặc Cloud / Hugging Face Space)"
    ).strip().rstrip("/")

    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("##### Thông số dữ liệu hiện tại")
    st.markdown("- **Tập huấn luyện:** 9.215 mẫu")
    st.markdown("- **Tập kiểm thử:** 2.000 mẫu")
    st.markdown("- **Số lớp ý định:** 5 nhãn")
    st.markdown("  - `GREETING` (Chào hỏi)")
    st.markdown("  - `TRADE` (Giao dịch)")
    st.markdown("  - `INFO` (Hỏi thông tin)")
    st.markdown("  - `HOSTILE` (Khiêu khích)")
    st.markdown("  - `UNKNOWN` (Ngoại luồng)")
    st.markdown("---")
    st.caption(f"Kiến trúc: Client-Server (HTTP REST API)")


# ==============================================================================
# GIAO DIỆN CHÍNH (KHÔNG SỬ DỤNG ICON Ở TIÊU ĐỀ VÀ PHÂN HẠNG)
# ==============================================================================
st.title("Hệ thống Nhận diện Ý định NPC")

tab_chat, tab_benchmark = st.tabs(["Trò chuyện NPC", "Hiệu năng và Đối sánh"])

# ==============================================================================
# TAB 1: TRÒ CHUYỆN VỚI NPC (GỌI QUA FASTAPI)
# ==============================================================================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    available_models, fetch_err = fetch_available_models(api_base_url)

    if fetch_err == "CONNECTION_ERROR":
        st.error(
            f"Không thể kết nối tới FastAPI server tại `{api_base_url}`. "
            "Vui lòng bật server backend trước bằng lệnh:\n\n"
            "```bash\nuvicorn restapi.app:app --port 8000\n```"
        )
        selected_model_name = None
    elif fetch_err == "TIMEOUT_ERROR":
        st.warning(f"Quá thời gian kết nối tới `{api_base_url}`. Vui lòng kiểm tra lại server.")
        selected_model_name = None
    elif fetch_err:
        st.error(f"Lỗi khi tải danh sách mô hình từ API: {fetch_err}")
        selected_model_name = None
    elif not available_models:
        st.warning("API server không trả về mô hình nào khả dụng.")
        selected_model_name = None
    else:
        col_sel, col_info = st.columns([3, 2])
        with col_sel:
            selected_model_name = st.selectbox(
                label="Chọn mô hình suy luận:",
                options=available_models,
                index=0
            )
        with col_info:
            st.caption(
                f"Trạng thái API: Đã kết nối | "
                f"Mô hình: **{selected_model_name}**"
            )

    st.markdown("---")

    # Hiển thị lịch sử trò chuyện
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "meta" in msg:
                    st.caption(msg["meta"])

    # Ô nhập liệu chat
    user_input = st.chat_input("Nhập tin nhắn gửi tới NPC...")

    if user_input:
        if not selected_model_name:
            st.warning(
                "Chưa kết nối được tới API server. Vui lòng khởi chạy: "
                "`uvicorn restapi.app:app --port 8000`"
            )
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            pred_intent, latency, gen_err = call_generate_api(
                api_base_url, selected_model_name, user_input
            )

            if gen_err == "CONNECTION_ERROR":
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "Mất kết nối tới FastAPI server. "
                        "Vui lòng kiểm tra lại lệnh `uvicorn restapi.app:app --port 8000`."
                    )
                })
            elif gen_err:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Không thể xử lý yêu cầu: {gen_err}"
                })
            else:
                meta_info = (
                    f"Mô hình: {selected_model_name} | "
                    f"API Round-trip: {latency:.2f} ms"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Ý định nhận diện:** `{pred_intent}`",
                    "meta": meta_info
                })
            st.rerun()

# ==============================================================================
# TAB 2: BẢNG ĐIỀU KHIỂN HIỆU NĂNG VÀ ĐỐI SÁNH (BENCHMARK DASHBOARD)
# ==============================================================================
with tab_benchmark:
    st.subheader("Báo cáo đối sánh các mô hình trên tập kiểm thử độc lập")
    st.caption("Tập kiểm thử chuẩn gồm 2.000 mẫu, bao gồm 70% cấu trúc câu chưa từng thấy trong tập huấn luyện.")

    col_btn, col_time = st.columns([2, 3])
    with col_btn:
        if st.button("Chạy lại đánh giá toàn diện"):
            with st.spinner("Đang kiểm thử đồng thời 5 mô hình trên 2.000 mẫu..."):
                from backend.benchmark import run_benchmark
                st.cache_data.clear()
                benchmark_data = run_benchmark(save_cache=True)
                st.success("Đã hoàn tất kiểm thử!")
                st.rerun()

    benchmark_data = load_benchmark_data()
    with col_time:
        st.caption(f"Thời gian cập nhật kết quả: {benchmark_data.get('timestamp', 'Mới nhất')}")

    # 1. Bảng số liệu tổng hợp
    st.markdown("#### Bảng đối sánh tổng thể")
    summary_raw = benchmark_data["summary"]

    formatted_rows = []
    chart_models = []
    chart_acc = []
    chart_f1 = []

    for rank_idx, row in enumerate(summary_raw, start=1):
        size_display = (
            f"{row['size_kb']/1024:.2f} MB" if row["size_kb"] > 1024 else f"{row['size_kb']} KB"
        )
        # Sử dụng số 1, 2, 3, 4, 5 để phân hạng (tuyệt đối không sử dụng icon)
        formatted_rows.append({
            "Hạng": rank_idx,
            "Mô hình": row["model_name"],
            "Framework": row["framework"],
            "Độ chính xác (%)": f"{row['accuracy']:.2f}%",
            "Macro F1 (%)": f"{row['macro_f1']:.2f}%",
            "Weighted F1 (%)": f"{row['weighted_f1']:.2f}%",
            "Độ trễ trung bình": f"{row['latency_ms']:.4f} ms",
            "Dung lượng lưu trữ": size_display
        })

        chart_models.append(row["model_name"])
        chart_acc.append(row["accuracy"])
        chart_f1.append(row["macro_f1"])

    df_summary = pd.DataFrame(formatted_rows)
    st.dataframe(df_summary, width="stretch", hide_index=True)

    st.markdown("---")

    # 2. Biểu đồ cột so sánh Accuracy và Macro F1
    st.markdown("#### Biểu đồ so sánh Độ chính xác giữa các mô hình")
    df_chart = pd.DataFrame({
        "Mô hình": chart_models,
        "Độ chính xác (Accuracy %)": chart_acc,
        "Macro F1 (%)": chart_f1
    }).set_index("Mô hình")

    st.bar_chart(df_chart, height=350)

    st.markdown("---")

    # 3. Ma trận nhầm lẫn (Confusion Matrix)
    st.markdown("#### Ma trận nhầm lẫn chi tiết")
    model_choices = [r["model_name"] for r in summary_raw]
    selected_cm_model = st.selectbox(
        label="Chọn mô hình để xem ma trận nhầm lẫn:",
        options=model_choices,
        index=0
    )

    if selected_cm_model in benchmark_data["confusion_matrices"]:
        cm_matrix = np.array(benchmark_data["confusion_matrices"][selected_cm_model])
        df_cm = pd.DataFrame(
            cm_matrix,
            index=[f"Thực tế: {name}" for name in INTENT_NAMES],
            columns=[f"Dự đoán: {name}" for name in INTENT_NAMES]
        )
        st.dataframe(df_cm, width="stretch")

    st.markdown("---")

    # 4. Phân tích các mẫu dự đoán sai tiêu biểu (Error Analysis)
    st.markdown("#### Phân tích các trường hợp dự đoán sai tiêu biểu")
    errors = benchmark_data["error_samples"].get(selected_cm_model, [])

    if not errors:
        st.info("Mô hình này không có trường hợp nào dự đoán sai trong số mẫu được trích xuất.")
    else:
        df_errors = pd.DataFrame([
            {
                "STT": idx + 1,
                "Câu truy vấn": err["text"],
                "Ý định thực tế": err["true_intent"],
                "Ý định dự đoán": err["pred_intent"]
            }
            for idx, err in enumerate(errors)
        ])
        st.dataframe(df_errors, width="stretch", hide_index=True)
