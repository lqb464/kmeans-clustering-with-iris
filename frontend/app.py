from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Clustomer", layout="wide")
API_URL = os.getenv("CLUSTOMER_API_URL", "http://127.0.0.1:8000")

st.title("Clustomer")
st.caption("Phân khúc hành vi khách hàng để hỗ trợ ra quyết định CRM")
st.info(
    "Các phân khúc hỗ trợ lập kế hoạch chiến dịch; chúng không dự đoán phản ứng nhân quả "
    "và không tự động quyết định cách xử lý khách hàng."
)

tab_score, tab_portfolio, tab_governance = st.tabs(
    ["Chấm điểm giao dịch", "Danh mục khách hàng", "Quản trị"]
)

with tab_score:
    cutoff = st.date_input("Mốc chấm điểm", value=pd.Timestamp("2010-10-31")).isoformat()
    upload = st.file_uploader("Tải lên tệp CSV giao dịch gốc", type="csv")
    if upload and st.button("Gán phân khúc", type="primary"):
        frame = pd.read_csv(upload)
        response = requests.post(
            f"{API_URL}/predict",
            json={"cutoff": cutoff, "transactions": frame.to_dict(orient="records")},
            timeout=120,
        )
        if response.ok:
            body = response.json()
            result = pd.DataFrame(body["predictions"])
            st.success(
                f"Đã chấm điểm {len(result):,} khách hàng bằng mô hình {body['model_version']}"
            )
            for warning in body.get("warnings", []):
                st.warning(warning)
            st.dataframe(result, width="stretch")
            st.download_button(
                "Tải kết quả dự đoán",
                result.to_csv(index=False),
                "clustomer_predictions.csv",
                "text/csv",
            )
        else:
            st.error(response.text)

with tab_portfolio:
    default_path = "outputs/customer_segments.csv"
    if os.path.exists(default_path):
        portfolio = pd.read_csv(default_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Khách hàng", f"{len(portfolio):,}")
        c2.metric("Phân khúc", portfolio["SegmentName"].nunique())
        c3.metric("Cần rà soát thủ công", f"{portfolio['ManualReview'].mean():.1%}")
        counts = (
            portfolio["SegmentName"]
            .value_counts()
            .sort_values()
            .rename_axis("Phân khúc")
            .reset_index(name="Khách hàng")
        )
        chart = px.bar(
            counts,
            x="Khách hàng",
            y="Phân khúc",
            orientation="h",
            text="Khách hàng",
            title="Phân bố khách hàng theo phân khúc",
        )
        chart.update_traces(textposition="outside")
        chart.update_layout(height=300, xaxis_title="Khách hàng", yaxis_title=None)
        st.plotly_chart(chart, width="stretch")
        st.dataframe(portfolio.loc[portfolio["ManualReview"].astype(bool)], width="stretch")
    else:
        st.warning("Hãy chạy quy trình huấn luyện để tạo kết quả gán phân khúc cho danh mục.")

with tab_governance:
    st.subheader("Phạm vi và biện pháp kiểm soát")
    st.markdown("""
    - Quần thể: khách hàng đã định danh tại Vương quốc Anh với giao dịch mua hoàn tất và giá trị dương.
    - Đầu vào: Recency, Frequency và Monetary, được tính tại một mốc thời gian xác định.
    - Rà soát thủ công: 5% lượt gán có độ bất định cao nhất.
    - Giới hạn đã biết: dữ liệu lịch sử chỉ bao phủ một năm; khách hàng thiếu mã bị loại; phân khúc không ước lượng tác động của chiến dịch.
    - Giám sát: lược đồ, PSI đặc trưng, tỷ trọng phân khúc, độ bất định và kết quả chiến dịch thực tế.
    """)
    try:
        st.json(requests.get(f"{API_URL}/model", timeout=10).json())
    except requests.RequestException:
        st.warning("Hiện không thể truy cập metadata của API.")
