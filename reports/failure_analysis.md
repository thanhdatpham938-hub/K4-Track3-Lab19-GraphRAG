# Phân Tích Ca Lỗi — Flat RAG vs GraphRAG

**Học viên:** Phạm Thành Đạt
**Khóa học:** AICB-K34 · Track 3: GraphRAG

**Bối cảnh:** phân tích dưới đây dựa trên kết quả thực chạy 25 câu Golden Dataset (`outputs/graphrag_eval_results.csv`) trên đồ thị 135 node / 93 edge, corpus 200 chunk. Chi tiết thiết lập benchmark và bảng điểm tổng hợp xem tại [`technical_defense.md`](technical_defense.md) mục 4.

---

## Ca lỗi 1: Không quan sát được "Flat RAG thất bại — GraphRAG thành công"

**Đây là một kết luận trung thực cần nêu rõ, không phải một chỗ trống bỏ sót.**

### Triệu chứng
Trong 25/25 câu hỏi, không có trường hợp nào GraphRAG cho điểm cao hơn rõ rệt so với Flat RAG. `graph_supernode_events = 0` ở **toàn bộ 25/25 câu** — nghĩa là graph traversal gần như không thu được cạnh liên quan nào.

### Truy vết nguyên nhân gốc rễ

Dùng cây quyết định dựa trên `diagnostics` của `retrieve_graph_context(question, return_debug=True)`:

```python
g = retrieve_graph_context(question, return_debug=True)
g["diagnostics"]["reason"]            # "NO_SEED" neu khong match duoc seed nao
g["diagnostics"]["matched_seeds"]     # seed nao da match, id gi
g["diagnostics"]["collected_edges"]   # so canh thu duoc
g["diagnostics"]["supernode_events"]  # node nao bi cat, bac bao nhieu
```

- `NO_SEED` → lỗi ở **seed matching** (fuzzy threshold 0.66 quá chặt, hoặc thực thể chưa từng được trích xuất).
- Có seed nhưng `collected_edges` thấp → lỗi ở **extraction** (thiếu cạnh).
- Có nhiều cạnh nhưng trả lời sai → lỗi ở **linearize/generation** (cạnh đúng bị cắt bởi `MAX_GRAPH_CONTEXT_CHARS`, hoặc generator không suy luận nối chuỗi được).

Áp dụng vào dữ liệu thật: với 24/25 câu, cả hai pipeline hội tụ về cùng kết luận "insufficient evidence" **vì cùng một lý do gốc** — 25 câu Golden được soạn từ một slice 5000 dòng đầu của dataset gốc, nhưng lần chạy này chỉ trích xuất 200/3,000 chunk từ một mẫu **1,500 bài lấy ngẫu nhiên**. Các bài báo cụ thể mà Golden Dataset tham chiếu (Amazon/Cohere, AMD/AWS, Google Cloud Next '23...) hầu như chắc chắn không nằm trong 200 chunk đó. Do đó phần lớn câu hỏi rơi vào nhánh `NO_SEED` hoặc match được seed nhưng `collected_edges` gần như 0 — **không có sự phân kỳ kiến trúc nào để so sánh**, vì cả hai pipeline cùng thiếu đúng một nguồn dữ liệu.

### Kết luận & đề xuất khắc phục
Đây chính xác là hệ quả đã cảnh báo trước ở mục 5a của `technical_defense.md`: *"GraphRAG chỉ hoàn vốn khi... đủ lớn"* — ở scale 200 chunk, đồ thị chưa đủ dày để tạo ra chuỗi quan hệ multi-hop nào cho GraphRAG khai thác. **Muốn quan sát được ca "GraphRAG thắng rõ rệt" cần chạy lại với `EXTRACTION_MAX_CHUNKS` đủ lớn để phủ đúng phạm vi 5000 dòng mà golden set tham chiếu** — hiện bị chặn bởi giới hạn TPD của Groq free tier (xem đề xuất scale ở `technical_defense.md` mục 5c: async worker queue + near-dedup để giảm chi phí LLM, cho phép tăng `EXTRACTION_MAX_CHUNKS` mà không vượt ngân sách token).

---

## Ca lỗi 2: Cả hai cùng sai một phần — `G5000-48`

**Câu hỏi DUY NHẤT trong 25 câu có đủ context nguồn thật** (nhóm `multi-hop`):

> *"Combine the three Snowflake-related records to summarize Snowflake's role in the 2023 cloud/data ecosystem: what AI integration, market recognition, and demand signal are present?"*

### Root-cause analysis

**Reference answer** cần tổng hợp 3 fact độc lập:
- (a) H2O AI Cloud tích hợp vào Snowflake Manufacturing Data Cloud
- (b) Snowflake vào danh sách CRN Big Data 100
- (c) Snowflake vượt ước tính quý nhờ nhu cầu dữ liệu tăng (theo báo cáo Reuters)

**Cả Flat RAG và GraphRAG đều chỉ tìm được 2/3 fact — (a) và (b)** — đúng, có trích dẫn `chunk_id` chính xác:

> *"AI Integration: Snowflake launched the Manufacturing Data Cloud, which includes H2O AI Cloud as a pre-built solution [chunk_id=5d162210c58b91f6a091::c0000]. Market Recognition: Snowflake was named one of the 'coolest big data system and cloud platform companies' in the CRN Big Data 100 [chunk_id=8f13d5e3ad34b056f423::c0000]."*

Cả hai **cùng bỏ sót fact (c)**. Truy nguyên gốc rễ: bài báo Reuters về "demand signal" đơn giản là không nằm trong 200 chunk đã trích xuất — đây là **giới hạn dữ liệu (corpus coverage), không phải giới hạn của kiến trúc retrieval**. Cả hai pipeline hành xử "đúng" theo nghĩa hẹp: không bịa fact (c) ra khi không có nguồn.

### Khác biệt thú vị giữa hai nhánh — điểm mấu chốt của ca lỗi này

Flat RAG im lặng bỏ qua fact (c):
> *"Demand Signal: The context does not provide specific evidence regarding demand signals for Snowflake."*

GraphRAG thì **chủ động trích dẫn một cạnh không liên quan** — demand signal của công ty `6sense` (một công ty hoàn toàn khác) — kèm `chunk_id` thật:
> *"Demand Signal: The provided context does not contain specific demand signals for Snowflake; it only mentions demand signals for 6sense (Forbes 2023 Cloud 100) [chunk_id=778d4aed58d3a78c6c58::c0000]."*

Đây là hành vi **đúng về mặt an toàn** (không bịa đặt, có trích dẫn, chủ động nói rõ dữ liệu đó không phải về Snowflake) — cả `flat_faithfulness` và `graph_faithfulness` đều được Judge chấm bằng nhau (3/5). Nhưng nó **lộ ra một rủi ro kiến trúc thật**: graph traversal đã kéo về một cạnh cùng loại quan hệ (`demand signal`) nhưng khác chủ thể, và generator phải tự phân biệt đúng/sai ngữ cảnh. Với `ANSWER_SYSTEM` prompt hiện tại (yêu cầu nghiêm ngặt "chỉ trả lời từ context, trích dẫn provenance") thì model đã xử lý đúng — nhưng nếu prompt kém chặt chẽ hơn, đây chính là điểm dễ phát sinh **context confusion**: gán nhầm dữ liệu của công ty này sang công ty khác. Về bản chất, đây cùng một lớp rủi ro với lỗi coreference phân tích ở `technical_defense.md` mục 1 — chỉ khác là xảy ra ở tầng retrieval (graph traversal kéo nhầm cạnh liên quan-nhưng-sai-chủ-thể) thay vì ở tầng trích xuất (gán nhầm đại từ).

### Đề xuất khắc phục
Seed matching / graph traversal nên áp thêm ràng buộc **entity-type-aware relevance scoring** khi mở rộng BFS ra khỏi node seed, để hạ ưu tiên (hoặc loại bỏ) các cạnh cùng loại quan hệ nhưng thuộc về node không nằm trong seed set ban đầu — thay vì để BFS tự do kéo về bất kỳ cạnh nào cùng nhãn quan hệ. Một lựa chọn khác nhẹ hơn: yêu cầu `ANSWER_SYSTEM` gắn nhãn rõ ràng "not directly about the queried entity" mỗi khi trích dẫn cạnh không thuộc node seed trực tiếp, để giảm nguy cơ người đọc hiểu nhầm cạnh đó là bằng chứng chính.
