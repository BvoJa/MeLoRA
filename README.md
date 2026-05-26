# MELoRA Demo for CS221

Đây là mã nguồn demo cho bài toán fine-tuning hiệu quả tham số bằng **LoRA** và **MELoRA** trên các tác vụ NLP tiếng Việt và benchmark NLU. Thư mục này tập trung vào PhoBERT cho **Vietnamese Sentiment Analysis** trên dataset `uitnlp/vietnamese_students_feedback`, đồng thời giữ lại script GLUE và LLaMA từ repo gốc.


MELoRA là phương pháp mở rộng LoRA bằng cách đóng băng trọng số của pretrained model và huấn luyện một nhóm mini-LoRA. Thay vì chỉ dùng một adapter low-rank, MELoRA dùng nhiều mini adapter để tăng khả năng tổng quát hóa với số tham số huấn luyện nhỏ.

## Method Overview

<div align="center">
  <img src="./figs/method.png">
</div>

## Cấu trúc chính

| File | Mục đích |
| --- | --- |
| `peft-0.5.0/` | Bản PEFT tùy chỉnh có hỗ trợ MELoRA. Cần cài đặt editable trước khi chạy. |
| `run_vietnamese_students_feedback_phobert.py` | Script huấn luyện/evaluate/predict PhoBERT cho Sentiment Analysis trên Vietnamese Students Feedback. |
| `vietnamese_students_feedback_phobert.sh` | Script bash cấu hình sẵn thí nghiệm PhoBERT + MELoRA cho dataset `uitnlp/vietnamese_students_feedback`. |
| `run_glue_lora.py`, `glue_finetune.sh` | Script LoRA/MELoRA cho GLUE benchmark. |
| `llama_finetune.py`, `llama_finetune.sh` | Script instruction tuning với LLaMA. |
| `utils/` | Helper callback, prompter và module tiện ích. |
| `templates/` | Prompt templates dùng cho instruction tuning. |
| `figs/method.png` | Hình minh họa phương pháp MELoRA. |
| `requirements.txt` | Thư viện Python cần cài đặt. |

## Cài đặt môi trường

Khuyến nghị dùng Python 3.10.

```bash
conda create -n MELoRA python=3.10
conda activate MELoRA

pip install torch==2.0.1
pip install -r requirements.txt

cd peft-0.5.0
pip install -e .
cd ..
```

Nếu chạy trên Kaggle/Colab và cần tạo môi trường conda, có thể cài `condacolab` trước rồi restart kernel theo hướng dẫn của notebook. Dataset `uitnlp/vietnamese_students_feedback` dùng Hugging Face dataset script, vì vậy `requirements.txt` đã pin `datasets==2.19.2` để tránh lỗi `Dataset scripts are no longer supported`.

Nếu chạy các model/dataset trên Hugging Face lần đầu, máy cần kết nối internet để tải model và dataset về cache.

## Cách chạy nhanh demo

Các script bash hiện đang đặt:

- `WANDB_MODE=offline`: log W&B ở chế độ offline.
- `seed=42`: cố định seed để tái lập kết quả.
- `mode=base`: LoRA baseline.
- `mode=me` hoặc `mode=melora`: MELoRA.
- `rank=8, l_num=1`: cấu hình LoRA baseline trong đa số script encoder.
- `rank=8, l_num=2`: cấu hình MELoRA trong đa số script encoder.
- `target_modules="query value"`: gắn adapter vào các module attention query/value của RoBERTa/PhoBERT.

Chạy từ thư mục hiện tại:

```bash
bash vietnamese_students_feedback_phobert.sh
```

Script trên sẽ chạy MELoRA cho Sentiment Analysis trên Vietnamese Students Feedback. Kết quả được ghi vào thư mục output tương ứng, ví dụ:

```text
./phobert-vsfc-sentiment/<run_name>/model
```

## Danh sách script thí nghiệm

| Script | Tác vụ | Model mặc định | Dataset | Cấu hình chính |
| --- | --- | --- | --- | --- |
| `vietnamese_students_feedback_phobert.sh` | Vietnamese student feedback sentiment classification | `vinai/phobert-base` | `uitnlp/vietnamese_students_feedback` | 20 epochs, batch 64, max length 256, Macro F1 |
| `glue_finetune.sh` | GLUE NLU benchmark | `FacebookAI/roberta-base` | GLUE | task-specific epochs/batch/max length/metric |
| `llama_finetune.sh` | Instruction tuning | `meta-llama/Llama-2-7b-hf` | template/data trong script Python | LoRA/MELoRA cho causal language modeling |

## Lệnh chạy từng thí nghiệm

```bash
# Vietnamese students feedback sentiment classification
bash vietnamese_students_feedback_phobert.sh

# GLUE benchmark
bash glue_finetune.sh

# Instruction tuning với LLaMA
bash llama_finetune.sh
```

## Tùy chỉnh thí nghiệm

Trước khi chạy, có thể sửa trực tiếp trong file `.sh`:

- `model_name_or_path`: đổi model Hugging Face hoặc đường dẫn model local.
- `dataset_name`: đổi dataset Hugging Face.
- `text_column`, `label_column`: đổi cột input/label cho bài toán classification.
- `num_train_epochs`, `learning_rate`, `batch_size`: cấu hình huấn luyện.
- `max_seq_length`: độ dài tối đa sau khi tokenize.
- `rank`: rank của LoRA hoặc rank của từng mini-LoRA.
- `l_num`: số lượng mini-LoRA khi dùng MELoRA.
- `wandb_project`: tên project W&B. Mặc định trong script là `project_name`.
- `CUDA_VISIBLE_DEVICES`: GPU dùng để chạy, nếu cần chỉ định GPU thủ công.

Với Vietnamese Students Feedback, file `vietnamese_students_feedback_phobert.sh` hiện đang chạy MELoRA:

```bash
run "me" "8" "2"
```

Nếu muốn chạy LoRA baseline, bỏ comment dòng:

```bash
# run "base" "8" "1"
```

Với `llama_finetune.sh`, cần đảm bảo có quyền truy cập model `meta-llama/Llama-2-7b-hf` hoặc thay `--base_model` bằng model local/phù hợp với máy demo.

## Output

Mỗi lần chạy tạo một thư mục riêng theo `run_name`. Thư mục này thường gồm:

- `model/`: checkpoint/model tốt nhất được lưu bởi Hugging Face Trainer.
- `log/`: log training, nếu script có truyền `--logging_dir`.
- `test_predictions.csv`: dự đoán trên test set cho script Vietnamese Students Feedback.
- log W&B offline, nếu `--report_to wandb` được bật.

Metric chính của demo Sentiment Analysis là **Macro F1** và **Accuracy**. GLUE dùng metric riêng theo task, ví dụ `pearson` cho STS-B và `matthews_correlation` cho CoLA.


## Thanks

Code được phát triển dựa trên:

- [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- [huggingface/peft](https://github.com/huggingface/peft)
- [huggingface/transformers](https://github.com/huggingface/transformers)

## Cite

Nếu sử dụng method/code này, vui lòng cite:

```bibtex
@inproceedings{ren-etal-2024-melora,
    title = "{MEL}o{RA}: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning",
    author = "Ren, Pengjie  and
      Shi, Chengshun  and
      Wu, Shiguang  and
      Zhang, Mengqi  and
      Ren, Zhaochun  and
      de Rijke, Maarten  and
      Chen, Zhumin  and
      Pei, Jiahuan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.168/",
    doi = "10.18653/v1/2024.acl-long.168",
    pages = "3052--3064",
    abstract = "Parameter-efficient fine-tuning (PEFT) is a popular method for tailoring pre-trained large language models (LLMs), especially as the models' scale and the diversity of tasks increase. Low-rank adaptation (LoRA) is based on the idea that the adaptation process is intrinsically low-dimensional, i.e., significant model changes can be represented with relatively few parameters. However, decreasing the rank encounters challenges with generalization errors for specific tasks when compared to full-parameter fine-tuning. We present MELoRA, a mini-ensemble low-rank adapters that uses fewer trainable parameters while maintaining a higher rank, thereby offering improved performance potential.The core idea is to freeze original pretrained weights and train a group of mini LoRAs with only a small number of parameters. This can capture a significant degree of diversity among mini LoRAs, thus promoting better generalization ability. We conduct a theoretical analysis and empirical studies on various NLP tasks. Our experimental results show that, compared to LoRA, MELoRA achieves better performance with 8 times fewer trainable parameters on natural language understanding tasks and 36 times fewer trainable parameters on instruction following tasks, which demonstrates the effectiveness of MELoRA."
}
```
