# Suy Ngẫm & Kế Hoạch Đồ Án — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Phạm Thành Đạt
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

Phần thuyết minh kỹ thuật (10 câu hỏi) xem tại [`technical_defense.md`](technical_defense.md); phân tích ca lỗi xem tại [`failure_analysis.md`](failure_analysis.md).

---

## 1. Mapping Bài giảng vào Code

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

## 2. Quá trình Debugging & Bài học

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

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

**Tên đồ án:** Ứng dụng nhắc uống thuốc cho người thân + Chatbot hỏi đáp về thuốc và sức khỏe người dùng.

### a) Bài toán có cần GraphRAG không?

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

### b) Cấu trúc Node & Relation dự kiến

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

### c) Chiến lược Entity Resolution

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

### d) Chiến lược Super-node

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

### e) Lộ trình triển khai đề xuất

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
| Mức độ hiểu bài giảng GraphRAG | 4 | Giải thích được cơ chế và đánh đổi của từng module (`technical_defense.md`), áp dụng được sang domain khác (mục 3) — nhưng chưa tự tay viết lại từ đầu nên chưa chắc nội hóa hết |
| Khả năng kiểm soát AI Coding Agent | 4 | Xem `technical_defense.md` mục 5b — 4 đề xuất đã từ chối kèm lý do kỹ thuật; đồng thời cũng phải sửa nhiều bug do chính agent để lại (bug pool bị bypass ở mục 2) — cho thấy cần kiểm tra kỹ output của agent, không tin tưởng mù quáng |
| Chất lượng đồ thị tri thức xây dựng | 2 | Trung thực: chỉ 135 node / 93 edge / bậc cao nhất 6 — quá thưa để thể hiện được super-node hay entity resolution tự nhiên (phải hạ ngưỡng để demo, xem `technical_defense.md` mục 2–3). Nguyên nhân là giới hạn TPD của Groq free tier buộc giảm `EXTRACTION_MAX_CHUNKS` xuống 200 |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết được 3 lớp lỗi độc lập (0-triple do model bị gỡ, rate-limit TPD, rò `<think>`) tới tận nguyên nhân gốc thay vì dừng ở triệu chứng — xem mục 2 |
