# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Thành Đạt
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Môi trường thực thi:** Local Jupyter (Windows 11), Python 3.10.20 (venv qua `uv`), Neo4j AuraDB Free
**LLM Extraction/Generation:** Groq `openai/gpt-oss-120b`
**LLM-as-a-Judge:** OpenAI `gpt-4o-mini`
**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine qua FAISS `IndexFlatIP`)

### Thông số dữ liệu thực tế của lần chạy này

| Chỉ số | Giá trị |
|---|---|
| Dataset | `HackerNoon/tech-company-news-data-dump` (gated, stream qua `HF_TOKEN`) |
| Dung lượng stream về | 25.0 MB (`.cache/hackernoon_subset.csv`) |
| Số bài thô | 20,907 |
| Sau exact dedup (SHA-1 `title+text`) | 18,860 (loại 2,047 bài ≈ 9.8%) |
| Sau Scale Guard `LAB_MAX_ARTICLES` | 1,500 |
| Số chunk (220 từ, overlap 40) | `LAB_MAX_CHUNKS` = 3,000 (đạt trần) |
| Số chunk đưa qua LLM trích xuất | `EXTRACTION_MAX_CHUNKS` = **200** (giảm từ 400 của spec — xem ghi chú bên dưới) |
| Golden Dataset | 25 câu — 2 `factoid`, 12 `multi-hop`, 11 `cross-doc` |
| **Node trong Neo4j** | **135** (`Company`/`Person`/`Technology`) |
| **Edge trong Neo4j** | **93**, `invalid_provenance_edges = 0` ✅ |
| **Triple trích xuất được** | **64** triple từ 200 chunk (192/200 chunk có kết quả; phần còn lại bị model rate-limit trong lúc chạy, xem Phần 2 mục 2) |
| **Bậc (degree) cao nhất trong đồ thị** | **6** (`SigenStor`) — thấp hơn xa ngưỡng `SUPER_NODE_DEGREE=100` vì corpus demo nhỏ |
| **LLM Judge — điểm trung bình** | Comprehensiveness 1.04 · Faithfulness 1.08 · Multi-hop 1.04 (thang 1–5, cả Flat lẫn GraphRAG) |

> **Ghi chú về cột dữ liệu:** dataset thực tế có schema `companyName, companyUrl, published_at, url, title, main_image, description` — **không có** cột `text/content/body`. Nội dung bài nằm ở cột `description`. Đây là điểm phải sửa `pick_col()` trong Module 1 (chi tiết ở Phần 2 mục 2).

> **Ghi chú về sai lệch so với Scale Guard của đề bài:** spec đặt `EXTRACTION_MAX_CHUNKS = 400`, nhưng lần chạy đầu với 400 chunk đã làm cạn hạn mức **200,000 token/ngày (TPD)** của Groq free tier ngay giữa giai đoạn extraction — 59/100 batch bị lỗi 429, chỉ thu được 58 triple và bảng audit Entity Resolution rỗng. Đây là một ràng buộc hạ tầng thật, không phải lỗi logic. Ba biện pháp đã áp dụng: (1) **model pool tự động failover** khi một model cạn TPD (mỗi model có bucket riêng → ngân sách 600k thay vì 200k), (2) **checkpoint sau mỗi batch** để cạn quota giữa chừng không mất công đã chạy, (3) giảm `EXTRACTION_MAX_CHUNKS` xuống **200** để tổng nhu cầu (~250k token) nằm gọn trong ngân sách và chừa đủ token cho giai đoạn đánh giá. Đánh đổi: đồ thị nhỏ hơn, nên các chỉ số phụ thuộc mật độ đồ thị (đặc biệt là bậc super-node) sẽ thấp hơn kỳ vọng của đề — được phân tích thẳng thắn ở Phần 1 mục 3.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*

- **Ví dụ từ dữ liệu (`chunk_id = b9987134e0e47e507473::c0000`):** Chunk gốc nói về chính sách bảo mật của một tổ chức (nội dung công khai không nêu tên tổ chức ngay trong chunk), chứa các đại từ `"we"` và `"your personal information"`. Hệ thống **không resolve** hai mention này — `resolved_text` giữ nguyên văn "we remain committed to upholding the highest standards when handling your personal information" — và ghi cả hai vào `unresolved_mentions = ["we", "your personal information"]`.
- **Hiện tượng:** đây **không phải một ca sai** mà là ca **conservative rule hoạt động đúng như thiết kế** — antecedent của "we" không xuất hiện tường minh trong chunk (chunk bị cắt mất câu giới thiệu tổ chức ở đoạn trước), nên hệ thống từ chối đoán thay vì suy diễn ra tên công ty từ metadata `companyName` của dòng dữ liệu (vốn không đáng tin cậy — xem ghi chú ở đầu báo cáo về việc `companyName` là công ty sở hữu trang HackerNoon, không phải chủ ngữ của bài viết). Đây chính xác là kiểu tình huống mà đề bài cảnh báo: nếu prompt kém conservative hơn, "we" rất dễ bị gán nhầm thành `companyName`.
- **Hậu quả nếu resolve sai (giả định phản chứng):** nếu hệ thống gán liều "we" → `companyName` của dòng dữ liệu, bước NER+RE ở Module 2 sẽ tạo ra một node `Company` sai và gắn quan hệ (ví dụ `USES`, `PARTNERED_WITH`) cho công ty đó dựa trên nội dung của một bài báo **không phải về công ty đó**. Vì `evidence` trong Neo4j trích nguyên câu gốc (nghe rất thuyết phục), lỗi này gần như không thể phát hiện chỉ bằng cách đọc lại đồ thị — phải đối chiếu ngược với `source_chunk_id` mới lộ ra.
- **Đánh đổi quan sát được:** 2/2 mention trong ví dụ này bị bỏ qua hoàn toàn (0% recall cho chunk này), đổi lấy 0% rủi ro tạo False Edge. Với domain tin tức công nghệ nói chung, đây là đánh đổi chấp nhận được vì một node/edge sai gây hại nhiều hơn một mention bị bỏ sót.

**Phân tích cấu trúc rủi ro (không phụ thuộc dữ liệu):**

Đặc thù dataset này khuếch đại rủi ro coreference theo một cách rất cụ thể: mỗi dòng có cột `companyName` (công ty sở hữu trang HackerNoon) **độc lập** với nội dung `description` (tin tức về công ty *khác*). Ví dụ dòng đầu tiên có `companyName = "01Synergy"` nhưng nội dung nói về `onsemi` và `Sineng Electric`. Nếu prompt coreference suy diễn "the company" → tên công ty ở metadata thay vì chủ ngữ trong câu, mọi triple trích ra sau đó sẽ bị gán sai chủ thể một cách hệ thống.

Notebook chống lại điều này bằng **conservative rule**: chỉ resolve khi antecedent xuất hiện rõ trong *cùng chunk*, mơ hồ thì giữ nguyên và log vào `unresolved_mentions`. Đánh đổi: giảm recall (nhiều đại từ không được resolve → mất cạnh) để đổi lấy precision (tránh False Edge). Với KG, đây là đánh đổi đúng — một cạnh sai độc hại hơn một cạnh thiếu, vì cạnh sai sẽ được GraphRAG trích dẫn kèm `evidence` trông rất thuyết phục.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*

**Ngưỡng đã dùng:** `threshold = 0.90` (cosine, trên embedding đã `normalize_embeddings=True` nên inner product = cosine), `top_k = 5` láng giềng ANN qua FAISS `IndexFlatIP`.

**Lý do chọn 0.90:** MiniLM-L6-v2 đẩy mọi tên riêng công ty công nghệ vào một vùng không gian rất hẹp — "Microsoft" và "Google" vẫn đạt cosine ~0.6–0.7 dù là hai thực thể hoàn toàn khác nhau. Ngưỡng thấp (0.80–0.85) sẽ gộp bừa các đối thủ cùng ngành. Ngưỡng 0.90 chỉ giữ lại các cặp gần như trùng chuỗi, đúng mục tiêu: bắt biến thể viết tên (`Microsoft Corp` vs `Microsoft Corporation`) chứ không phải bắt quan hệ ngữ nghĩa.

**Kiến trúc 3 tầng phòng thủ** (`build_resolution_map()`):

| Tầng | Cơ chế | Vai trò |
|---|---|---|
| 1 | `MANUAL_ALIASES` (ticker + tên phổ biến: `msft→Microsoft`, `googl→Google`, `aapl→Apple`…) | Bắt trường hợp vector **không thể** bắt: ticker `MSFT` và `Microsoft` có cosine thấp vì khác hoàn toàn về mặt chuỗi |
| 2 | FAISS ANN, cosine ≥ 0.90 | Sinh **candidate**, không tự quyết định |
| 3 | `merge_guard()` — strip hậu tố (`inc/corp/ltd/llc/plc/co/company`) rồi `SequenceMatcher ratio ≥ 0.72` | **Phủ quyết** candidate của tầng 2 |

Điểm mấu chốt kiến trúc: **vector chỉ được đề xuất, lexical mới được quyết định.** Union-Find (`UF`) chỉ `union(i,j)` khi guard trả `True`; mọi cặp bị chặn đều ghi `REJECT_GUARD` vào `entity_resolution_audit_df` để audit minh bạch.

**Kết quả thực tế — không có cặp nào vượt 0.85, chứ đừng nói 0.90.** Đây tự nó là một phát hiện quan trọng: chạy `build_resolution_map(raw_triples_df, threshold=0.90)` trên corpus 200-chunk (108 mention thực thể duy nhất) cho ra **0 audit row** — similarity cao nhất đo được toàn corpus chỉ là **0.644**. Để chứng minh cơ chế guard vẫn hoạt động đúng, tôi hạ ngưỡng xuống **0.45 chỉ cho lần chạy demo này** (không đổi khuyến nghị production 0.90), thu được 22 audit row — **toàn bộ đều là `REJECT_GUARD`, 0 `MERGE_VECTOR`**:

| Left | Right | Similarity | Decision | Lý do chặn |
|---|---|---|---|---|
| `high-speed fiber internet` | `optical fiber` | 0.644 | `REJECT_GUARD` | Cùng chủ đề hạ tầng mạng nhưng là hai khái niệm khác nhau (dịch vụ vs vật liệu); strip suffix không đổi, `SequenceMatcher` thấp |
| `Renoworks FastTrack` | `Renoworks API` | 0.635 | `REJECT_GUARD` | Cùng công ty `Renoworks` nhưng là hai **sản phẩm** khác nhau của công ty đó — gộp sẽ xóa mất sự phân biệt sản phẩm |
| `Aqara` | `AYYA` | 0.618 | `REJECT_GUARD` | Hai công ty smart-home hoàn toàn khác nhau, embedding chỉ bắt được "gần miền ngữ nghĩa" (IoT/nhà thông minh), không phải đồng nhất |
| `Samsung` | `Apple` | 0.575 | `REJECT_GUARD` | Ca kinh điển: hai đối thủ cùng ngành bị đẩy gần nhau trong không gian embedding — đúng loại lỗi mà ngưỡng 0.90 được thiết kế để ngăn |
| `Future Technology Group` | `Jobs for the Future` | 0.562 | `REJECT_GUARD` | Trùng cụm từ "Future" nhưng là hai tổ chức không liên quan |

**Ý nghĩa cho việc chọn ngưỡng:** kết quả này củng cố trực tiếp lý do chọn 0.90 ở trên — ngay cả ở dải similarity 0.45–0.65 (thấp hơn nhiều so với 0.90), **không một cặp nào trong dữ liệu thật đáng được gộp**. Nếu hạ ngưỡng production xuống mức này, hệ thống sẽ tạo ra hàng chục candidate nhiễu (như `Samsung`↔`Apple`) mà may mắn vẫn bị lexical guard chặn — nhưng đó là may mắn của dữ liệu này, không phải đảm bảo kiến trúc. Với domain có nhiều biến thể viết tên hơn (tên người, thương hiệu đa ngôn ngữ), ngưỡng thấp sẽ để lọt nhiều false positive hơn nữa mà guard không kịp chặn hết.

**Loại False Merge mà guard này ngăn được (phân tích thiết kế):**
- *Người trùng họ:* `Sam Altman` vs `Steve Altman` — cosine rất cao (cùng họ, cùng cấu trúc tên) nhưng `SequenceMatcher("sam altman","steve altman")` thấp hơn 0.72 → chặn. Nếu gộp, mọi phát ngôn của người này bị gán cho người kia.
- *Sản phẩm mang tên công ty:* `Apple Watch` vs `Apple` — sau khi strip suffix vẫn là hai chuỗi khác độ dài rõ rệt → ratio thấp → chặn. Nếu gộp, `Apple Watch [Technology]` biến mất và mọi cạnh của nó dồn vào node `Apple`, làm trầm trọng thêm vấn đề super-node.
- *Ngược lại, cặp cần gộp vẫn qua được:* `Microsoft Corp` vs `Microsoft Corporation` → strip suffix cho ra `microsoft` == `microsoft` → `merge_guard` trả `True` ngay ở nhánh so sánh bằng, không cần tới ngưỡng 0.72.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*

**Top 3 thực thể theo bậc (trên đồ thị thật: 135 node, 93 edge):**

| Hạng | Tên thực thể | Loại (Type) | Bậc (Degree) |
|------|--------------|-------------|--------------|
| 1 | SigenStor | Company | 6 |
| 2 | Future Technology Group | Company | 4 |
| 3 | SC Ventures / Aeris / Microsoft / Dalet / Apple / Amazon / Google | Company | 3 (đồng hạng) |

**Ghi chú quan trọng — không có super-node thật ở scale demo này:** bậc cao nhất chỉ **6**, thấp hơn xa ngưỡng `SUPER_NODE_DEGREE=100`. Chạy `test_supernode_policy()` ở ngưỡng production xác nhận: *"Top node degree (6) chưa vượt threshold (100) → cap chưa kích hoạt."* Đây là hệ quả trực tiếp của việc giảm `EXTRACTION_MAX_CHUNKS` xuống 200 (xem ghi chú Scale Guard đầu báo cáo) — 64 triple là quá ít để bất kỳ thực thể nào tích lũy bậc ba chữ số.

**Để chứng minh cơ chế cap thực sự hoạt động** (không chỉ tồn tại trên giấy), tôi gọi thêm `test_supernode_policy(degree_threshold=3)` — **chỉ hạ tham số của lệnh gọi test, không đổi hằng số `SUPER_NODE_DEGREE` toàn cục** (nên không ảnh hưởng đến kết quả retrieval/đánh giá đã chạy ở Phần 3–4). Kết quả: node `SigenStor` (degree=6) vượt ngưỡng demo=3 → giới hạn về `cap=50` → `assert len(edges) <= 50` **pass** → in `✅ Super-node cap OK (threshold=3, cap=50)`. Cơ chế đúng về logic, chỉ chưa có dữ liệu đủ lớn để tự nhiên kích hoạt ở ngưỡng thật.

**Cấu hình chính sách đã dùng:**

```python
SUPER_NODE_DEGREE      = 100    # nguong xac dinh super-node
SUPER_NODE_EDGE_CAP    = 50     # cap canh khi vuot nguong
GLOBAL_EDGE_CAP        = 250    # tran toan bo context
MAX_GRAPH_CONTEXT_CHARS = 14000 # tran ky tu sau khi linearize
```

Cơ chế trong `retrieve_graph_context()`: mỗi node khi pop khỏi BFS frontier đều được gọi `node_degree()`; nếu `degree > 100` thì `limit = min(edge_limit, 50)` và sự kiện được ghi vào `diagnostics["supernode_events"]`. Truy vấn `recent_edges()` sắp xếp `ORDER BY coalesce(r.published_date,'') DESC LIMIT $limit` — tức việc cắt tỉa được đẩy **xuống tầng database**, không phải lấy hết về Python rồi mới lọc. Đây là điểm quan trọng về hiệu năng: nếu lọc ở tầng ứng dụng, một node bậc 5,000 vẫn phải truyền 5,000 record qua mạng.

**Ưu điểm của Temporal Mitigation (ưu tiên cạnh mới nhất):**
1. **Chặn context explosion:** không có cap, một seed trúng `Microsoft` bậc vài nghìn sẽ nhồi toàn bộ token budget bằng cạnh nhiễu, đẩy thông tin liên quan ra khỏi cửa sổ ngữ cảnh.
2. **Phù hợp domain tin tức:** dataset là tin công nghệ 2023, câu hỏi Golden phần lớn hỏi về diễn biến gần — "mới nhất" tương quan mạnh với "liên quan nhất".
3. **Xác định (deterministic) và audit được:** cùng câu hỏi cho cùng tập cạnh, và `supernode_events` ghi lại chính xác node nào bị cắt, bậc bao nhiêu, cap còn bao nhiêu.

**Rủi ro tiềm ẩn:**
1. **Mù thông tin lịch sử:** câu hỏi kiểu "Microsoft mua lại công ty nào năm 2016" sẽ trượt nếu 50 cạnh mới nhất đều là 2023. Đây là lỗi *im lặng* — hệ thống trả lời "không đủ bằng chứng" chứ không báo rằng nó đã tự cắt mất dữ liệu.
2. **Thiên lệch theo tần suất đưa tin:** thực thể được báo chí nhắc nhiều dồn cạnh vào một dải thời gian hẹp, đẩy các quan hệ cũ nhưng quan trọng (ví dụ `FOUNDED`) ra ngoài — trong khi quan hệ `FOUNDED` gần như luôn là sự kiện cũ.
3. **Phụ thuộc chất lượng `published_date`:** `coalesce(r.published_date,'')` đẩy cạnh thiếu ngày xuống **cuối** danh sách sắp xếp DESC. Cạnh thiếu ngày gần như chắc chắn bị loại tại super-node. Sanity check `invalid_provenance_edges == 0` vì thế không chỉ để lấy điểm — nó là điều kiện tiên quyết để chính sách cắt tỉa này công bằng.
4. **Cắt tỉa cục bộ nhưng ảnh hưởng toàn cục:** cạnh bị cắt ở hop 1 khiến node láng giềng đó không bao giờ vào frontier, nên hop 2 mất luôn cả một nhánh — thiệt hại lớn hơn con số 50 gợi ý.

**Hướng cải thiện đề xuất:** thay cắt thuần theo thời gian bằng chiến lược lai — quota theo `relation type` (đảm bảo mỗi loại quan hệ giữ được vài cạnh, tránh `PARTNERED_WITH` nhiều lấn át `FOUNDED` hiếm), kết hợp `confidence` và độ liên quan của cạnh với câu hỏi, và khi câu hỏi có mốc thời gian thì lọc theo cửa sổ thời gian đó thay vì luôn lấy mới nhất.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

**Thiết lập so sánh (giữ biến kiểm soát):** cả hai nhánh dùng **cùng** embedding model, **cùng** generator (`openai/gpt-oss-120b`), **cùng** `ANSWER_SYSTEM` prompt. Khác biệt duy nhất là kiến trúc retrieval:

| | Flat RAG | Hybrid GraphRAG |
|---|---|---|
| Nguồn context | FAISS top-k=6 chunk | Subgraph BFS (`max_hops=2`) **+** FAISS top-k=4 chunk |
| Định dạng context | Văn bản thô | `=== GRAPH ===` (cạnh đã linearize kèm provenance) + `=== VECTOR ===` |
| Đơn vị truy hồi | Đoạn văn | Cạnh có `source_chunk_id`, `published_date`, `evidence` |

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, thang 1–5) — số liệu thật từ `outputs/graphrag_vs_flatrag_summary.csv`:

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|-------------------|----------|----------|---|-------------------|
| **Comprehensiveness (1–5)** | 1.04 | 1.04 | 0 | Gần như bằng nhau — xem giải thích "vì sao thấp" bên dưới |
| **Faithfulness (1–5)** | 1.08 | 1.08 | 0 | Gần như bằng nhau |
| **Multi-hop Reasoning (1–5)** | 1.04 | 1.04 | 0 | Gần như bằng nhau |
| **Latency trung bình (s)** | 2.46 | 1.91 | -0.55 | GraphRAG **nhanh hơn** trong sample này (ngược trực giác — lý do bên dưới) |
| **Token usage trung bình** | 776.6 | 682.4 | -94.2 | GraphRAG **rẻ hơn** trong sample này |

**Vì sao điểm chất lượng đồng loạt ~1.0/5 cho cả hai phương pháp — đây là phát hiện chính của benchmark này, không phải một con số vô nghĩa:**

25 câu Golden (mã `G5000-26` … `G5000-50`) được soạn từ một **slice 5000 dòng đầu** của dataset gốc (thấy rõ qua evidence gốc trong `data/graphrag_golden_50_first5000.csv`, ví dụ `"row 2532"`, `"row 3357"`). Nhưng theo Scale Guard của lab, `standardize_news()` lấy mẫu **ngẫu nhiên** `LAB_MAX_ARTICLES=1500` từ 18,860 bài sau dedup, rồi `EXTRACTION_MAX_CHUNKS=200` chỉ trích xuất 200/3,000 chunk đầu tiên của mẫu đó. Xác suất các bài báo cụ thể mà Golden Dataset tham chiếu (Amazon/Cohere, AMD/AWS, Google Cloud Next '23...) rơi vào đúng 200 chunk này là rất thấp — và thực tế đã xảy ra: **24/25 câu**, cả Flat RAG lẫn GraphRAG đều trả lời đúng đắn "ngữ cảnh không đủ bằng chứng" (`flat_answer` mẫu: *"The provided excerpts do not contain any information about Amazon's July AI-service expansion..."*). Judge chấm 1/5 cho cả hai vì **cả hai đều đúng** khi từ chối trả lời — không phải vì retrieval kém.

→ **Kết luận quan trọng:** benchmark này đo đúng "độ trung thực khi thiếu bằng chứng" (cả hai kiến trúc đều không bịa đặt — điểm cộng cho `ANSWER_SYSTEM` prompt), nhưng **không đo được** sự khác biệt kiến trúc Flat vs Graph, vì corpus quá nhỏ so với phạm vi golden set. Để benchmark có ý nghĩa thật, cần tăng `EXTRACTION_MAX_CHUNKS`/`LAB_MAX_ARTICLES` đủ lớn để phủ đúng phạm vi 5000 dòng mà golden set tham chiếu — bị chặn bởi giới hạn TPD của Groq free tier (phân tích chi tiết ở mục 5c).

**Vì sao GraphRAG lại NHANH HƠN và RẺ HƠN Flat RAG ở đây** (ngược với dự đoán ở mục 5a): sau khi vá lỗi rò `<think>` (model reasoning leak — xem Phần 2 mục 2), câu trả lời "insufficient evidence" rất ngắn cho cả hai nhánh. Với 200 chunk, `retrieve_graph_context()` hầu như luôn trả về `NO_SEED` hoặc rất ít cạnh (`graph_supernode_events=0` ở toàn bộ 25/25 câu — seed matching không tìm thấy entity nào đủ liên quan trong đồ thị 135-node), nên context đưa vào generator **ngắn hơn** context vector top-6 của Flat RAG — dẫn tới ít token sinh ra hơn. Đây là **artifact của corpus nhỏ**, không phải ưu điểm kiến trúc: một GraphRAG "rẻ hơn" vì tìm thấy quá ít thứ để nói không phải là thành công.

**Bảng theo nhóm câu hỏi:**

| Nhóm | Số câu | Flat (Comp/Faith/MH) | Graph (Comp/Faith/MH) | Nhận xét |
|---|---|---|---|---|
| `factoid` | 2 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | Cả 2 câu đều thiếu context nguồn |
| `multi-hop` | 12 | 1.08 / 1.17 / 1.08 | 1.08 / 1.17 / 1.08 | Câu `G5000-48` (xem ca lỗi #2) kéo điểm nhóm này lên nhẹ so với 1.0 tuyệt đối |
| `cross-doc` | 11 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | Đồng loạt "insufficient evidence" |

#### Phân tích 2 Ca lỗi Điển hình:

**1. Không quan sát được ca "Flat RAG thất bại — GraphRAG thành công" trong lần chạy này — và đây là một kết luận trung thực cần nêu rõ, không phải chỗ trống bỏ sót.**

Lý do trực tiếp: `graph_supernode_events = 0` ở **toàn bộ 25/25 câu**, nghĩa là graph traversal gần như không thu được cạnh liên quan nào (đồ thị 135 node quá thưa, seed matching thường xuyên miss). Với 24/25 câu, cả hai pipeline hội tụ về cùng kết luận "insufficient evidence" vì lý do giống hệt nhau (đã phân tích ở trên) — không có sự phân kỳ kiến trúc để so sánh. Đây chính xác là hệ quả đã cảnh báo trước ở mục 1.5a: *"GraphRAG chỉ hoàn vốn khi... đủ lớn"* — ở scale 200 chunk, đồ thị chưa đủ dày để tạo ra chuỗi quan hệ multi-hop nào cho GraphRAG khai thác. **Muốn quan sát được ca này cần chạy lại với `EXTRACTION_MAX_CHUNKS` đủ lớn để phủ đúng phạm vi golden set** (xem đề xuất mục 5c).

**2. Ca cả hai cùng sai một phần — `G5000-48` (multi-hop, câu hỏi DUY NHẤT trong 25 câu có đủ context nguồn):**
- *Câu hỏi:* "Combine the three Snowflake-related records to summarize Snowflake's role in the 2023 cloud/data ecosystem: what AI integration, market recognition, and demand signal are present?"
- *Reference:* cần tổng hợp 3 fact — (a) H2O AI Cloud tích hợp vào Snowflake Manufacturing Data Cloud, (b) Snowflake vào danh sách CRN Big Data 100, (c) Snowflake vượt ước tính quý nhờ nhu cầu dữ liệu tăng (theo Reuters).
- *Cả Flat RAG và GraphRAG đều chỉ tìm được 2/3 fact (a) và (b)* — đúng và có trích dẫn `chunk_id` chính xác — nhưng **cùng bỏ sót fact (c)**. Nguyên nhân gốc: bài báo Reuters về "demand signal" không nằm trong 200 chunk đã trích xuất — **giới hạn dữ liệu, không phải giới hạn retrieval**.
- *Khác biệt thú vị giữa hai nhánh:* GraphRAG, thay vì im lặng bỏ qua fact (c) như Flat RAG, **chủ động trích dẫn một cạnh không liên quan** — demand signal của công ty `6sense` (một công ty khác hoàn toàn) — kèm `chunk_id` thật, rồi viết "*context does not contain specific demand signals for Snowflake; it only mentions demand signals for 6sense*". Đây là hành vi đúng về mặt an toàn (không bịa, có trích dẫn), nhưng **lộ ra rủi ro kiến trúc thật**: graph traversal kéo về một cạnh cùng loại quan hệ (`demand signal`) nhưng khác chủ thể, và generator phải tự phân biệt — nếu prompt kém chặt chẽ hơn `ANSWER_SYSTEM` hiện tại, đây chính là điểm dễ phát sinh **context confusion** (gán nhầm dữ liệu của công ty này sang công ty khác), tương tự lớp lỗi coreference đã phân tích ở mục 1.
- *Đề xuất khắc phục:* seed matching nên áp thêm ràng buộc entity-type-aware relevance scoring khi mở rộng BFS, để tránh kéo về cạnh cùng loại quan hệ nhưng khác chủ thể vào context — hoặc yêu cầu generator gắn nhãn rõ "not directly about the queried entity" khi trích dẫn cạnh từ node khác node seed.

**Gợi ý truy vết nguyên nhân gốc rễ (dùng khi điền 2 ca trên):**
```python
g = retrieve_graph_context(question, return_debug=True)
g["diagnostics"]["reason"]            # "NO_SEED" neu khong match duoc seed nao
g["diagnostics"]["matched_seeds"]     # seed nao da match, id gi
g["diagnostics"]["collected_edges"]   # so canh thu duoc
g["diagnostics"]["supernode_events"]  # node nao bi cat, bac bao nhieu
```
Cây quyết định: `NO_SEED` → lỗi ở **seed matching** (fuzzy threshold 0.66 quá chặt, hoặc thực thể chưa từng được trích xuất). Có seed nhưng `collected_edges` thấp → lỗi ở **extraction** (thiếu cạnh). Có nhiều cạnh nhưng trả lời sai → lỗi ở **linearize/generation** (cạnh đúng bị cắt bởi `MAX_GRAPH_CONTEXT_CHARS`, hoặc generator không suy luận nối chuỗi được).

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

#### a) Đánh đổi Quality vs Cost vs Latency

**Chi phí một lần (indexing) — đây mới là chi phí thật của GraphRAG, số liệu đo trực tiếp trên lần chạy 200 chunk:**

| Giai đoạn | Số lệnh gọi LLM | Ghi chú |
|---|---|---|
| Coreference (200 chunk, batch 5) | 40 | ~9–13s/batch khi chạy trơn tru |
| NER + RE extraction (200 chunk, batch 4) | 50 | ~9–13s/batch |
| **Tổng indexing GraphRAG** | **90** | |
| Đánh giá 25 câu × (1 seed-extract + 1 flat-answer + 1 graph-answer) | 75 | Groq. Judge (OpenAI `gpt-4o-mini`) tính riêng, không tính vào TPD Groq |
| **Tổng lệnh gọi Groq cả phiên** | **~165** | |
| **Flat RAG indexing** | **0 lệnh gọi LLM** | chỉ embedding 3,000 chunk (~vài chục giây trên CPU) |

**Bằng chứng thực nghiệm cho việc "chi phí index tăng tuyến tính":** chính 165 lệnh gọi này đã **ăn hết hạn mức 200,000 token/ngày (TPD)** của 2/3 model trong pool (`openai/gpt-oss-120b` và `openai/gpt-oss-20b` đều báo `rate_limit_exceeded` giữa phiên chạy — xem log lỗi ở Phần 2 mục 2). Với chỉ 200 chunk và 25 câu hỏi, một tài khoản Groq free tier đã chạm trần. Quy chiếu tuyến tính: `EXTRACTION_MAX_CHUNKS=400` theo đúng spec đề bài sẽ cần ~180 lệnh gọi chỉ riêng cho indexing — thực tế đã kiểm chứng: lần chạy đầu tiên với 400 chunk bị rate-limit **59/100 batch extraction**, chỉ trích được 58/dự kiến ~400 triple. Đây là lý do phải giảm xuống 200 và xây `MODEL_POOL` failover (chi tiết Phần 2 mục 2) — **không phải chọn tùy tiện, mà là giới hạn hạ tầng đo được trực tiếp.**

Quy chiếu tiếp: toàn bộ 18,860 bài sau dedup sẽ cần hàng chục nghìn lệnh gọi LLM — **chi phí index tăng tuyến tính theo dữ liệu, trong khi chi phí truy vấn thì không**. GraphRAG chỉ hoàn vốn khi số truy vấn đủ lớn để chia đều chi phí index.

**Chi phí lặp lại (per-query):** GraphRAG đắt hơn Flat RAG ở mọi câu hỏi vì thêm 1 lệnh gọi LLM trích seed, N vòng Cypher round-trip (mỗi node BFS gọi `node_degree()` + `recent_edges()` — đây là **N+1 query problem**, điểm tối ưu rõ ràng nhất), và context dài hơn do gộp cả graph lẫn vector. Đổi lại là khả năng nối chuỗi quan hệ mà top-k vector không làm được.

**Kết luận đánh đổi:** GraphRAG **không** thay thế Flat RAG. Với `factoid`, Flat RAG cho chất lượng tương đương với chi phí thấp hơn nhiều. Kiến trúc production hợp lý là **query router**: phân loại câu hỏi trước, chỉ đẩy `multi-hop`/`cross-doc` qua nhánh graph.

#### b) Đề xuất của AI Coding Agent mà tôi TỪ CHỐI

**Từ chối 1 — Pairwise cosine $O(N^2)$ cho Entity Resolution.**
Cách làm hiển nhiên là so mọi cặp thực thể. Với ~vài nghìn mention, ma trận similarity đã là hàng chục triệu ô; scale lên toàn dataset thì tràn RAM. Tôi giữ **FAISS ANN + `top_k=5`**, đưa độ phức tạp về gần tuyến tính. Đây cũng đúng yêu cầu Challenge A trong notebook ("Không chấp nhận pairwise cosine $O(N^2)$ trên toàn dataset").

**Từ chối 2 — Bỏ Lexical Guard, tin hoàn toàn vào vector similarity.**
Lập luận "cosine 0.90 là đủ cao rồi" nghe hợp lý nhưng sai về bản chất: embedding đo **độ gần ngữ nghĩa**, không đo **đồng nhất thực thể (identity)**. `Apple` và `Apple Watch` gần nhau về ngữ nghĩa nhưng là hai thực thể khác nhau; gộp chúng làm hỏng đồ thị theo cách không thể phục hồi. Tôi giữ guard làm tầng phủ quyết.

**Từ chối 3 — `MERGE` từng dòng thay vì `UNWIND` theo batch.**
Vòng lặp `for row: session.run(MERGE ...)` dễ đọc hơn nhưng mỗi row là một network round-trip tới AuraDB. Với vài nghìn cạnh, latency mạng chiếm gần như toàn bộ thời gian. `UNWIND $rows AS row` với batch 1,000 gom lại thành vài request — đúng yêu cầu Module 2.

**Từ chối 4 — Nuốt exception để "pipeline chạy mượt".**
Đây là bài học đắt nhất của buổi lab (chi tiết ở Phần 2 mục 2). Pattern `try/except: continue` trong `run_coref()` và `run_extraction()` khiến lỗi cấu hình biến thành *dữ liệu rỗng* thay vì *thông báo lỗi*. Tôi đã **thêm fail-fast** vào 3 vị trí thay vì để nguyên: smoke test Groq ngay sau khi định nghĩa wrapper, kiểm tra tỉ lệ batch coref thất bại, và `raise` nếu extraction cho ra 0 triple.

#### c) Giải pháp kiến trúc khi scale lên 350MB (~100,000 bài)

**Bottleneck đầu tiên: LLM extraction throughput.** Lần chạy này 400 chunk mất ~15 phút với 100 lệnh gọi tuần tự. 100,000 bài (~200,000 chunk) sẽ cần ~50,000 lệnh gọi — chạy tuần tự là hàng trăm giờ. Đây là nút thắt trước cả Neo4j lẫn embedding.

| Tầng | Vấn đề khi scale | Giải pháp |
|---|---|---|
| **Extraction** | Tuần tự, ~9s/batch | Async worker queue + concurrency có kiểm soát rate-limit; checkpoint theo `chunk_id` để resume; **near-dedup trước khi extract** (MinHash/LSH) vì tin tức repost rất nhiều — exact hash lần này đã loại 9.8%, near-dedup sẽ loại thêm đáng kể và giảm thẳng chi phí LLM |
| **Entity Resolution** | FAISS `IndexFlatIP` là exhaustive, $O(N)$ mỗi truy vấn | Chuyển sang **HNSW** (`IndexHNSWFlat`); thêm **blocking** theo ký tự đầu / token đầu để chỉ so trong khối; Union-Find vẫn giữ vì gần như tuyến tính |
| **Neo4j ingestion** | Batch 1,000 vẫn ổn, nhưng index phải có trước | Giữ `UNWIND`; đảm bảo unique constraint trên `(:Entity).id` **trước** khi insert (nếu không, mỗi `MERGE` là full scan); cân nhắc `neo4j-admin import` cho lần nạp đầu |
| **Truy vấn (super-node)** | Bậc node sẽ tăng hàng chục lần, `GLOBAL_EDGE_CAP=250` trở nên quá thô | Vật chất hóa `degree` thành property (tránh đếm lại mỗi lần); gom `node_degree` + `recent_edges` thành **một** Cypher để bỏ N+1 query; thêm quota theo relation type |
| **Câu hỏi vĩ mô** | BFS local không trả lời được câu hỏi kiểu "xu hướng đầu tư AI 2023" | **Community detection** (Leiden/Louvain) + sinh community report, thêm **query router** chọn giữa local search và global search |

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|--------|------------------------|-----------------------------|
| **Exact Dedup** | M1 | `standardize_news()` — SHA-1 trên `title+text` đã chuẩn hóa | Loại 2,047/20,907 bài (9.8%). Chỉ bắt trùng **tuyệt đối**; bài repost đổi tiêu đề vẫn lọt → cần near-dedup (Challenge A) |
| **Rolling-window Chunking** | M1 | `chunk_text(size=220, overlap=40)` | Overlap 40 từ giữ ngữ cảnh qua ranh giới chunk — quan trọng vì conservative coref chỉ resolve **trong cùng chunk**, chunk cắt ẩu là mất antecedent |
| **Conservative Coreference** | M1 | `resolve_coref_batch()`, `run_coref()` | Ưu tiên precision hơn recall; ambiguity → giữ nguyên + log `unresolved_mentions`. Đúng hướng cho KG |
| **Schema & Allowlist Guard** | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, filter trong `run_extraction()` | Lọc **sau** khi LLM trả lời: mọi relation ngoài 8 loại cho phép bị loại thẳng. Đây là ràng buộc cứng ở tầng code, không phụ thuộc việc LLM có tuân thủ prompt hay không |
| **Edge Provenance** | M2 | `bulk_insert_edges()` + `graph_checks()` | 4 thuộc tính bắt buộc: `source_chunk_id`, `published_date`, `evidence`, `confidence`. `MERGE ... {source_chunk_id: row.source_chunk_id}` khiến cùng quan hệ từ hai chunk khác nhau là **hai cạnh riêng** — cố ý, để giữ được nhiều nguồn dẫn chứng độc lập |
| **Bulk Cypher Ingestion** | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()`, `batches(size=1000)` | `UNWIND $rows AS row`, nhóm theo label/relation type vì Cypher không cho tham số hóa tên label |
| **Entity Resolution & Union-Find** | M3 | `build_resolution_map()`, `merge_guard()`, `UF` | 3 tầng: manual alias → vector candidate → lexical veto. Union-Find gom cụm bắc cầu; canonical name chọn theo tần suất xuất hiện cao nhất |
| **Audit Trail** | M3 | `entity_resolution_audit_df`, `show_resolution_audit()` | Mọi quyết định gộp/từ chối đều ghi lại (`MERGE_MANUAL`/`MERGE_VECTOR`/`REJECT_GUARD`) — không có bảng này thì không thể debug được False Merge |
| **Flat RAG Baseline** | M4 | `build_flat_index()`, `retrieve_flat_context(k=6)` | FAISS `IndexFlatIP` + embedding chuẩn hóa = cosine similarity |
| **Seed Extraction & Fuzzy Fallback** | M4 | `extract_seeds()`, `match_seeds(fuzzy_threshold=0.66)` | Exact match trên `name_norm`/`aliases_norm` trước, không thấy mới fallback vector. Ngưỡng 0.66 thấp hơn 0.90 của ER — hợp lý vì đây là *tra cứu*, sai thì chỉ tốn context, không làm hỏng đồ thị |
| **BFS + Super-node Degree Cap** | M4 | `retrieve_graph_context()`, `recent_edges()`, `node_degree()` | Cắt tỉa đẩy xuống Cypher (`ORDER BY published_date DESC LIMIT`), không lọc ở tầng Python |
| **Linearization + Provenance** | M4 | `textualize()` | Mỗi cạnh thành một dòng có `date=`, `chunk=`, `evidence=` → LLM có thể trích dẫn nguồn thay vì bịa |
| **Hybrid Context** | M4 | `answer_graph_rag()` | Ghép `=== GRAPH ===` + `=== VECTOR ===`; graph cho quan hệ, vector cho sắc thái văn bản gốc |
| **LLM-as-a-Judge** | M5 | `judge_answer()`, `JUDGE_SYSTEM` | Dùng model **khác** (OpenAI `gpt-4o-mini`) với generator (Groq) để giảm thiên vị tự chấm; `temperature=0`; điểm bị kẹp `max(1, min(5, ...))` |
| **Benchmark Reporting** | M5 | `comparison_table()`, export CSV | Nhóm theo `group` để lộ ra chỗ GraphRAG thật sự thắng, thay vì chỉ nhìn trung bình toàn cục |

---

### 2. Quá trình Debugging & Bài học

**Lỗi kỹ thuật phức tạp nhất: pipeline "chạy thành công" trong 27 phút nhưng cho ra 0 triple.**

*Triệu chứng:* Cell 1.7 (coref) chạy 749s và cell 2.1 (extraction) chạy 936s — cả hai đều báo hoàn tất 100%, progress bar chạy đều đặn ~9s/batch, **không một dòng lỗi nào**. Nhưng cell 2.2 (Entity Resolution) vỡ với `AttributeError: 'DataFrame' object has no attribute 'source_raw'`.

*Quá trình truy vết:*
1. Thông báo lỗi trỏ vào Entity Resolution, nhưng cell đó không sai — nó chỉ đang đọc một DataFrame **rỗng**. Lỗi thật nằm ở cell trước đó.
2. Kiểm tra output cell 2.1: `Empty DataFrame, Columns: [], Index: []` — extraction chạy đủ 100 batch nhưng không sinh ra triple nào.
3. Tách một batch ra chạy riêng ngoài notebook để xem LLM trả về gì → `groq.APIConnectionError` với `Illegal header value b'Bearer '`.
4. Gọi trực tiếp Groq API → `404: The model 'llama-3.3-70b-versatile' does not exist or you do not have access to it`.
5. Liệt kê model khả dụng trên tài khoản → model trong `.env` (lấy từ README của đề bài) **đã bị Groq gỡ bỏ**.

*Nguyên nhân gốc rễ:* Model đã ngừng phục vụ. Nhưng điều biến một lỗi cấu hình 5 giây thành một cuộc điều tra 30 phút là **pattern nuốt exception**:

```python
try:
    obj, _ = extract_batch(batch)
except Exception as e:
    errors.append({"start": start, "error": str(e)})
    continue   # <-- pipeline di tiep nhu khong co gi xay ra
```

Cộng thêm `groq_chat(max_retries=4)` với backoff `2**attempt`, mỗi batch thất bại tiêu tốn đúng ~9 giây sleep — **trùng khớp với tốc độ của một batch thành công**, nên progress bar trông hoàn toàn bình thường. Lỗi được ghi vào `extraction_errors_df` nhưng biến này không được in ra ở đâu cả.

*Cách xử lý:*
1. Đổi `GROQ_MODEL` sang `openai/gpt-oss-120b` (đã xác minh JSON mode hoạt động trước khi chạy lại).
2. Kiểm chứng luôn OpenAI judge `gpt-4o-mini` **trước** khi chạy pipeline dài — tránh chạy 30 phút rồi mới chết ở cell cuối vì lỗi tương tự.
3. Thêm **3 chốt fail-fast** vào notebook: smoke test Groq ngay sau khi định nghĩa LLM wrapper (sai key/model là biết trong 2 giây), đếm batch coref thất bại và `raise` nếu toàn bộ fail, in số triple + số batch lỗi và `raise` nếu extraction cho ra 0 triple.

**Bài học rút ra:** trong pipeline nhiều tầng chạy dài, `except: continue` là phản pattern nguy hiểm — nó biến *lỗi* thành *dữ liệu rỗng*, và dữ liệu rỗng thì lan truyền âm thầm cho tới khi vỡ ở một nơi hoàn toàn không liên quan. Nguyên tắc: **thất bại toàn phần phải ồn ào; chỉ thất bại cục bộ mới được im lặng.** Một smoke test 2 giây ở đầu pipeline đáng giá hơn 27 phút chạy vô ích.

**Ca lỗi phức tạp thứ hai: rate-limit TPD của Groq free tier gây gián đoạn pipeline nhiều lần liên tiếp.**

Sau khi đổi sang model còn hoạt động (`openai/gpt-oss-120b`), pipeline chạy được một đoạn rồi liên tục dính `429 rate_limit_exceeded` với thông báo *"tokens per day (TPD): Limit 200000, Used 199522"*. Đây không phải lỗi code mà là **giới hạn hạ tầng thật** của gói miễn phí — mỗi model có ngân sách 200k token/ngày riêng, và pipeline này (coref + extraction + đánh giá 25 câu × nhiều lượt) tiêu thụ vượt xa mức đó chỉ trong một model.

Chuỗi khắc phục theo từng lớp:
1. **Model pool + failover:** thay vì phụ thuộc 1 model, dùng 3 model (`gpt-oss-120b` → `gpt-oss-20b` → `qwen3.6-27b`), mỗi model có bucket TPD riêng → ngân sách hiệu dụng ~600k thay vì 200k.
2. **Phát hiện thêm 1 bug ẩn trong chính cơ chế failover:** hàm `generate_answer()` gọi `groq_chat(..., model=GROQ_MODEL)` với **model tường minh**, trong khi logic pool ban đầu viết `candidates = [model] if model else [pool]` — nghĩa là hễ có `model=` là **bỏ qua hoàn toàn pool**. Bug này khiến toàn bộ giai đoạn đánh giá vẫn crash 429 dù đã có 3 model dự phòng. Sửa: `groq_chat()` giờ luôn thử `model` chỉ định trước, nhưng vẫn failover qua các model còn lại trong pool nếu model đó báo hết TPD, thay vì dừng ngay.
3. **Checkpoint sau mỗi batch/câu hỏi** (`coref_partial.pkl`, `triples_partial.pkl`, `graphrag_eval_checkpoint.csv`): cạn quota giữa chừng không còn mất công đã chạy — chạy lại tự động resume đúng chỗ dừng.
4. **Giảm `EXTRACTION_MAX_CHUNKS` từ 400 xuống 200** để tổng nhu cầu nằm gọn trong ngân sách, chừa đủ token cho giai đoạn đánh giá.

**Ca lỗi thứ ba, phát hiện muộn nhất — rò rỉ chain-of-thought vào câu trả lời:** sau khi có dữ liệu đánh giá đầu tiên, phát hiện nhiều `flat_answer`/`graph_answer` bắt đầu bằng `<think>...</think>` — các model reasoning (`gpt-oss`, `qwen3.6`) mặc định in cả quá trình suy luận vào `content` nếu không tắt tường minh. Điều này vừa làm nhiễu output, vừa **thổi phồng token usage và latency** một cách giả tạo (so sánh: trước khi vá, `flat_latency_s` trung bình ~10.5s/câu; sau khi vá chỉ còn ~2.5s/câu — cùng một câu hỏi, cùng model). Khắc phục bằng tham số `reasoning_effort` của Groq API — nhưng mỗi họ model dùng tên giá trị khác nhau (`gpt-oss` chỉ chấp nhận `low/medium/high`, `qwen` chỉ chấp nhận `none/default`, thử sai giá trị là lỗi 400 ngay lập tức) — nên phải map riêng theo tiền tố tên model, cộng thêm một lớp `_strip_think()` bằng regex làm lưới an toàn cuối cùng.

**Bài học chung cho cả 3 ca:** lỗi hạ tầng bên ngoài (model bị gỡ, rate limit, model rò reasoning) đều là những thứ **không thể lường trước từ việc đọc code một lần**, mà chỉ lộ ra khi chạy thật với dữ liệu thật. Thiết kế pipeline chống chịu (checkpoint, failover, fail-fast) quan trọng ngang với thiết kế thuật toán đúng — một pipeline "đúng về logic" nhưng không chống chịu được rate-limit thực tế thì vẫn không chạy được trong 2 giờ lab.

**Các lỗi khác đã gặp và xử lý:**

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `KeyError: Missing one of columns: ['text','content','article','body','story']` | Dataset thật dùng cột `description`, không có cột nào trong danh sách ứng viên của `pick_col()` | Thêm `"description"` vào danh sách ứng viên |
| `DatasetNotFoundError: gated dataset` | `HF_TOKEN` trong `.env` vẫn là placeholder `hf_...`; dataset lại là gated | Tạo token thật + bấm "Agree and access" trên trang dataset |
| Kernel treo, không log | Chạy nbconvert với `kernel_name=python3` → trỏ vào Python hệ thống thiếu thư viện, không phải venv | Chỉ định đúng kernel `lab19-graphrag` của venv 3.10 |
| `UnicodeEncodeError: 'charmap' codec` | Console Windows dùng cp1258, không in được tiêu đề cell tiếng Việt | `sys.stdout.reconfigure(encoding="utf-8")` |
| `AttributeError: 'str' object has no attribute 'get'` khi extraction | Model yếu hơn trong pool (`gpt-oss-20b`/`qwen`) đôi khi trả JSON đúng cú pháp nhưng sai schema (`items` chứa chuỗi thay vì object) | Parse phòng thủ: kiểm tra `isinstance(item, dict)` trước khi gọi `.get()`, ghi log vào `extraction_errors_df` thay vì crash |
| `ServiceUnavailable: Unable to retrieve routing information` (Neo4j) | Driver AuraDB bị idle quá lâu (~33 phút do một cell chạy chậm bất thường vì tranh chấp tài nguyên) → hết hạn routing table | Chạy lại với kernel mới (driver mới); AuraDB free tier không giữ kết nối idle lâu |

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

**Tên đồ án:** Ứng dụng nhắc uống thuốc cho người thân + Chatbot hỏi đáp về thuốc và sức khỏe người dùng.

#### a) Bài toán có cần GraphRAG không?

**Câu trả lời: Cần — nhưng chỉ cho một phần, và phần đó không được dùng KG trích xuất bằng LLM.**

Phân loại truy vấn của ứng dụng thành 3 nhóm, mỗi nhóm cần một kiến trúc khác nhau:

| Nhóm truy vấn | Ví dụ | Kiến trúc phù hợp | Lý do |
|---|---|---|---|
| **A. Lịch cá nhân** | "Mẹ tôi tối nay uống thuốc gì?", "Còn mấy viên Amlodipin?" | **Query CSDL trực tiếp** — không RAG | Dữ liệu có cấu trúc sẵn trong app. Đưa qua LLM chỉ thêm độ trễ và rủi ro sai lệch |
| **B. Kiến thức chung** | "Huyết áp cao nên ăn uống thế nào?", "Tập thể dục bao nhiêu là đủ?" | **Flat RAG** | Câu trả lời nằm gọn trong 1–2 đoạn tài liệu. Không cần nối quan hệ |
| **C. Suy luận liên thuốc — cá nhân hóa** | "Mẹ tôi đang uống Amlodipin và Metformin, giờ bác sĩ kê thêm thuốc này có sao không?" | **GraphRAG** | Bắt buộc nối chuỗi: `Thuốc mới → Hoạt chất → TƯƠNG_TÁC_VỚI → Hoạt chất → Thuốc mẹ đang uống`, đồng thời đối chiếu với bệnh nền và tiền sử dị ứng của **chính người dùng đó** |

Nhóm C chính là lý do GraphRAG hoàn vốn. Flat RAG **về nguyên tắc không giải được** nhóm này: thông tin về thuốc A và thuốc B nằm ở hai tài liệu khác nhau, và không tài liệu nào chứa sẵn câu "A tương tác với B trong bối cảnh bệnh nhân có bệnh nền X". Vector search top-k sẽ lấy về hai tờ hướng dẫn sử dụng riêng lẻ rồi để LLM tự suy diễn — đúng kiểu tình huống dễ sinh ảo giác nhất.

> **⚠️ Ràng buộc an toàn — quyết định kiến trúc quan trọng nhất của đồ án này:**
> Bài lab dùng LLM để trích xuất đồ thị từ văn bản tự do. **Với dữ liệu thuốc, tuyệt đối không làm như vậy.** Một cạnh `INTERACTS_WITH` sai do LLM trích nhầm sẽ trực tiếp gây hại cho người dùng thật, khác hẳn một cạnh sai trong đồ thị tin tức công nghệ. Kiến trúc phải tách đôi:
> - **Sự kiện y khoa** (tương tác thuốc, chống chỉ định, liều tối đa) → nạp từ **nguồn có thẩm quyền, có cấu trúc sẵn**: DrugBank, RxNorm, ATC, Dược thư Quốc gia Việt Nam, dữ liệu đăng ký thuốc của Cục Quản lý Dược. Đây là **ETL**, không phải LLM extraction.
> - **Nội dung tham khảo** (bài viết sức khỏe, hướng dẫn lối sống) → mới dùng Flat RAG + LLM.
>
> Ngoài ra, chatbot cần chốt chặn sản phẩm: luôn trích dẫn nguồn, không bao giờ tự đề xuất đổi liều/ngưng thuốc, và mọi câu thuộc nhóm C đều kết thúc bằng khuyến nghị xác nhận với bác sĩ/dược sĩ.

#### b) Cấu trúc Node & Relation dự kiến

Giữ allowlist **hẹp** như bài học từ lab (3 node type / 8 relation là đủ để có đồ thị hữu ích):

**Nodes:**
```
Drug            (Thuốc thương mại: Panadol, Amlor)   — key: so_dang_ky
ActiveIngredient(Hoạt chất: Paracetamol, Amlodipin)  — key: RxNorm CUI / ATC
Condition       (Bệnh: Tăng huyết áp, Đái tháo đường)— key: ICD-10
Symptom         (Triệu chứng / tác dụng phụ)
Patient         (Người thân được chăm sóc)           — dữ liệu riêng tư, tách schema
```

**Relations:**
```
Drug            -CONTAINS->            ActiveIngredient
ActiveIngredient-INTERACTS_WITH->      ActiveIngredient   (co: severity, mechanism)
Drug            -TREATS->              Condition
Drug            -CONTRAINDICATED_FOR-> Condition
Drug            -CAUSES_SIDE_EFFECT->  Symptom
Patient         -TAKES->               Drug               (co: lieu, tan suat, tu ngay)
Patient         -HAS_CONDITION->       Condition
Patient         -ALLERGIC_TO->         ActiveIngredient
```

**Hai điểm thiết kế mấu chốt:**

1. **Tương tác gắn ở tầng `ActiveIngredient`, không phải `Drug`.** Panadol và Efferalgan đều chứa Paracetamol; nếu gắn tương tác ở tầng thuốc thương mại thì phải nhân bản cạnh cho hàng trăm biệt dược và chắc chắn sẽ sót. Truy vấn luôn đi qua `Drug -CONTAINS-> ActiveIngredient -INTERACTS_WITH->`. Đây cũng chính là **quá liều Paracetamol do uống hai biệt dược cùng lúc** — một trong những tai nạn thuốc phổ biến nhất, và đồ thị thiết kế đúng sẽ phát hiện được.

2. **Provenance bắt buộc như trong lab, nhưng đổi trường cho hợp domain:** thay `published_date/source_chunk_id` bằng `source` (DrugBank/Dược thư), `severity` (major/moderate/minor), `evidence_level`, `last_reviewed`. Sanity check tương đương `invalid_provenance_edges == 0`: **không cạnh `INTERACTS_WITH` nào được thiếu `source` và `severity`** — vì UI phải hiển thị được "cảnh báo này dựa trên nguồn nào".

#### c) Chiến lược Entity Resolution

**Nguyên tắc: domain này CÓ định danh chuẩn, nên gần như không dùng vector similarity để quyết định danh tính.** Đây là điểm khác biệt lớn nhất so với bài lab (tin tức không có mã định danh nên buộc phải đoán bằng vector + guard).

| Tầng | Cơ chế | Ghi chú |
|---|---|---|
| 1 | Khớp **chính xác** theo mã: số đăng ký thuốc, RxNorm CUI, mã ATC | Nguồn chân lý. Xong ở đây là dừng |
| 2 | Bảng alias **do người soạn và duyệt**: biệt dược → hoạt chất (`Panadol → Paracetamol`), tên thương mại theo vùng | Tương đương `MANUAL_ALIASES` trong lab, nhưng bắt buộc có người review |
| 3 | Fuzzy matching | **Chỉ dùng cho ô nhập liệu của người dùng** (gõ tên thuốc để thêm vào lịch), và **luôn hiển thị danh sách gợi ý để người dùng tự xác nhận** — không bao giờ tự động chọn |

**Lý do cấm auto-merge bằng vector trong nhánh y khoa** — bài học `REJECT_GUARD` từ lab áp dụng ở đây với hậu quả nghiêm trọng hơn nhiều:
- `Losartan` vs `Lovastatin`: chuỗi rất giống (`SequenceMatcher` cao), nhưng một thuốc hạ huyết áp, một thuốc hạ mỡ máu. Guard theo chuỗi của lab **sẽ gộp nhầm**.
- `Paracetamol` vs `Paracetamol + Codein`: là hai thuốc khác nhau (loại sau chứa opioid, có chống chỉ định riêng). Cosine gần như 1.0.
- `Metformin 500mg` vs `Metformin 1000mg`: khác hàm lượng — với thuốc, hàm lượng là một phần của danh tính.

→ Kết luận: **thứ ở lab là "guard" thì ở đây phải là "cấm"**. Vector chỉ dùng để *gợi ý cho người dùng chọn*, không bao giờ để *hệ thống tự quyết*.

#### d) Chiến lược Super-node

**Hub dự kiến:** `Paracetamol` (có mặt trong hàng trăm biệt dược), `Tăng huyết áp` / `Đái tháo đường` (hàng trăm thuốc điều trị), các tá dược phổ biến.

**Chính sách "50 cạnh mới nhất" của lab là SAI cho domain này** — đây là điều chỉnh quan trọng nhất khi chuyển kiến trúc:
- Với tin tức, "mới nhất ≈ liên quan nhất". Với y khoa, **một tương tác thuốc phát hiện năm 1995 vẫn nguy hiểm y như năm 2024**. Cắt theo `last_reviewed DESC` có thể loại bỏ đúng cảnh báo cần hiển thị.

**Chính sách thay thế — cắt theo mức độ nghiêm trọng và theo loại quan hệ:**

```
1. KHONG BAO GIO cat canh: INTERACTS_WITH (severity=major),
                            CONTRAINDICATED_FOR, ALLERGIC_TO
   -> day la canh an toan, thieu mot canh la sai ca cau tra loi

2. Cat co quota theo relation type (tranh mot loai lan at loai khac):
   INTERACTS_WITH  moderate/minor -> uu tien theo severity
   CAUSES_SIDE_EFFECT             -> uu tien theo tan suat gap
   TREATS                         -> gioi han manh nhat

3. Thu hep pham vi truoc khi duyet do thi (quan trong nhat):
   Khong BFS mo tu node `Paracetamol`. Bat dau tu node `Patient`,
   chi lay cac thuoc benh nhan DANG uong -> tap seed chi vai hoat chat.
   Super-node gan nhu bien mat vi truy van luon duoc neo vao mot benh nhan cu the.
```

Điểm 3 là khác biệt kiến trúc căn bản: bài lab duyệt đồ thị mở từ thực thể trong câu hỏi, còn ứng dụng này luôn có **ngữ cảnh neo là một bệnh nhân cụ thể** với 3–10 thuốc. Không gian tìm kiếm nhỏ và có biên rõ ràng, nên vấn đề super-node được giải quyết ngay từ khâu thiết kế truy vấn thay vì phải chữa cháy bằng cắt tỉa.

#### e) Lộ trình triển khai đề xuất

| Giai đoạn | Nội dung | Ghi chú |
|---|---|---|
| 1 | Chức năng nhắc thuốc + CSDL lịch uống (nhóm A) | Chưa cần RAG. Đây là phần tạo giá trị sớm nhất |
| 2 | Flat RAG cho hỏi đáp sức khỏe chung (nhóm B) | Rẻ, nhanh, phủ phần lớn câu hỏi |
| 3 | Nạp KG thuốc từ nguồn có thẩm quyền (ETL, **không** LLM extraction) | DrugBank/RxNorm/Dược thư → Neo4j bằng `UNWIND` như lab |
| 4 | GraphRAG neo theo bệnh nhân cho nhóm C + guardrail an toàn | Chỉ mở khi KG đã được kiểm định |
| 5 | Query router phân loại A/B/C | Đúng bài học trade-off từ lab: không đẩy mọi câu qua nhánh đắt nhất |

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Giải thích được cơ chế và đánh đổi của từng module (Phần 1), áp dụng được sang domain khác (Phần 2 mục 3) — nhưng chưa tự tay viết lại từ đầu nên chưa chắc nội hóa hết |
| Khả năng kiểm soát AI Coding Agent | 4 | Xem Phần 1 mục 5b — 4 đề xuất đã từ chối kèm lý do kỹ thuật; đồng thời cũng phải sửa nhiều bug do chính agent để lại (bug pool bị bypass ở mục 2) — cho thấy cần kiểm tra kỹ output của agent, không tin tưởng mù quáng |
| Chất lượng đồ thị tri thức xây dựng | 2 | Trung thực: chỉ 135 node / 93 edge / bậc cao nhất 6 — quá thưa để thể hiện được super-node hay entity resolution tự nhiên (phải hạ ngưỡng để demo, xem Phần 1 mục 2–3). Nguyên nhân là giới hạn TPD của Groq free tier buộc giảm `EXTRACTION_MAX_CHUNKS` xuống 200 |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết được 3 lớp lỗi độc lập (0-triple do model bị gỡ, rate-limit TPD, rò `<think>`) tới tận nguyên nhân gốc thay vì dừng ở triệu chứng — xem Phần 2 mục 2 |
