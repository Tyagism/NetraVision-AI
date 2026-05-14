# Graph Report - .  (2026-05-08)

## Corpus Check
- Corpus is ~44,670 words - fits in a single context window. You may not need a graph.

## Summary
- 482 nodes · 664 edges · 48 communities (35 shown, 13 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 94 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Application|Core Application]]
- [[_COMMUNITY_Main Entry|Main Entry]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Alert Management|Alert Management]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Depth Estimation|Depth Estimation]]
- [[_COMMUNITY_TTS Engine|TTS Engine]]
- [[_COMMUNITY_Object Detection|Object Detection]]
- [[_COMMUNITY_Spatial Reasoning|Spatial Reasoning]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Parler-TTS Architecture|Parler-TTS Architecture]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_OCR Reader|OCR Reader]]
- [[_COMMUNITY_Priority Logic|Priority Logic]]

## God Nodes (most connected - your core abstractions)
1. `ParlerTTSForConditionalGeneration` - 28 edges
2. `AlertManager` - 27 edges
3. `NetraVisionAi` - 21 edges
4. `TTSEngine` - 21 edges
5. `ParlerTTSLogitsProcessor` - 20 edges
6. `ParlerTTSDecoderConfig` - 19 edges
7. `DepthEstimator` - 18 edges
8. `ParlerTTSConfig` - 18 edges
9. `ParlerTTSForCausalLM` - 14 edges
10. `AccuracyBenchmark` - 13 edges

## Surprising Connections (you probably didn't know these)
- `NetraVision AI Core` --implements--> `Visually Impaired Support`  [INFERRED]
  netra_vision_ai_core.py → README.md
- `TTS Engine` --supports--> `Indic Language Support`  [INFERRED]
  modules/tts_engine.py → README.md
- `main()` --calls--> `NetraVisionAi`  [INFERRED]
  main.py → netra_vision_ai_core.py
- `NetraVisionAi` --uses--> `Camera`  [INFERRED]
  netra_vision_ai_core.py → modules/camera.py
- `NetraVisionAi` --uses--> `ObjectDetector`  [INFERRED]
  netra_vision_ai_core.py → modules/detector.py

## Communities (48 total, 13 thin omitted)

### Community 0 - "Core Application"
Cohesion: 0.05
Nodes (43): DACConfig, LogitsProcessor, ModelOutput, ParlerTTSConfig, ParlerTTSDecoderConfig, r"""     This is the configuration class to store the configuration of a [`Parl, r"""     This is the configuration class to store the configuration of an [`Par, ParlerTTSLogitsProcessor (+35 more)

### Community 1 - "Main Entry"
Cohesion: 0.07
Nodes (24): IntEnum, Alert, AlertPriority, PriorityEngine, Create a LOW priority scene description alert., Alert priority levels., A single alert to be spoken., Evaluates spatial objects and generates prioritized alerts.     Uses cooldown t (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (10): apply_delay_pattern_mask(), generate(), ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration, Build a delayed pattern mask to the input_ids. Each codebook is offset by the pr, Apply a delay pattern mask to the decoder input ids, only preserving predictions, Prepares `decoder_input_ids` for generation with encoder-decoder models, Initializes input ids for generation, if necessary. (+2 more)

### Community 3 - "Alert Management"
Cohesion: 0.08
Nodes (13): AlertManager, Process alerts with natural pacing and summarization.          Instead of spea, Alias for process_alerts., Speak critical alert immediately — interrupt everything., Only keep alerts about NEW objects or CHANGED distances., Intelligent alert manager that produces calm, natural speech output., Extract position from alert message., Combine multiple alerts into one natural sentence.          Instead of: (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (23): convert_dataset_str_to_list(), DataCollatorEncodecWithPadding, DataCollatorParlerTTSWithPadding, load_multiple_datasets(), Data collator that will dynamically pad the inputs received to the longest seque, Data collator that will dynamically pad the inputs received.     Args:, main(), # NOTE: filtering is done at the end because in the `datasets` library, caching (+15 more)

### Community 5 - "Depth Estimation"
Cohesion: 0.09
Nodes (17): Get average depth within object bounding box, Generate a natural language scene description.         Used for the 'slow lane', Maps detected objects to spatial descriptions with depth, Map detected objects to spatial descriptions.                  Args:, SpatialMapper, AccuracyBenchmark, main(), NetraVisionAi - Comprehensive Accuracy & Performance Benchmarking Evaluates: Ob (+9 more)

### Community 6 - "TTS Engine"
Cohesion: 0.1
Nodes (14): Verify TTS engine is available., Initialize Piper voice model., Start the speech output thread., Background thread: processes speech queue., Add text to speech queue.         Args:             text: Text to speak, Actually speak the text using the selected engine., Speak using espeak-ng., Speak using Piper TTS (Python API - cross-platform). (+6 more)

### Community 7 - "Object Detection"
Cohesion: 0.09
Nodes (14): Camera, NetraVisionAi — Camera Module Handles frame capture from webcam / phone camera, Initialize camera.                  Args:             source: Camera index (0, Start camera capture in background thread, Continuously capture frames in background, Get the latest frame.                  Returns:             (frame, frame_cou, OCRReader, Get all readable text from frame as a single string.         Useful for "read e (+6 more)

### Community 8 - "Spatial Reasoning"
Cohesion: 0.11
Nodes (11): DepthEstimator, print_system_info(), NetraVisionAi — Depth Estimation Smart model selection + automatic GPU/CPU dete, Find local .pt weights file, Load model — tries local .pt first, then torch.hub, Load image preprocessing transform, Run warmup inference to initialize everything, Estimate depth from BGR frame.          Returns:             depth_map: same (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (15): apply_rotary_pos_emb(), _get_unpad_data(), ParlerTTSAttention, ParlerTTSFlashAttention2, ParlerTTSSdpaAttention, This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The, Rotates half the hidden dims of the input., Applies Rotary Position Embedding to the query and key tensors.      Args: (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (10): Scene Description using lightweight Vision Language Model Options: Moondream2,, SmolVLM is not available, Lightweight Vision Language Model for scene description.                  Args, Select best available device, Load moondream2 model, Load Moondream2 — best quality for size, Load SmolVLM — NOT AVAILABLE due to dependency conflicts, Generate scene description from BGR frame (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (5): DACModel, Decodes the given frames into an output audio waveform.          Note that the, # TODO: for now, no chunk length, Encodes the input audio waveform into discrete codes.          Args:, # TODO: for now, no chunk length

### Community 12 - "Parler-TTS Architecture"
Cohesion: 0.19
Nodes (7): ObjectDetector, NetraVisionAi — Object Detection Module YOLOv8-nano via ONNX Runtime, Preprocess frame for YOLO input, Parse YOLOv8 output format, Detect objects AND draw bounding boxes (for debugging/demo), Initialize YOLOv8-nano object detector.                  Args:             mo, Detect objects in frame.                  Args:             frame: BGR image

### Community 13 - "Community 13"
Cohesion: 0.23
Nodes (5): BaseStreamer, ParlerTTSStreamer, Flushes any remaining cache and appends the stop symbol., Put the new audio in the queue. If the stream is ending, also put a stop signal, Streamer that stores playback-ready audio in a queue, to be used by a downstream

### Community 14 - "Community 14"
Cohesion: 0.22
Nodes (11): Alert Manager, Camera Interface, Depth Estimation, Indic Language Support, NetraVision AI Core, Object Detection, Parler-TTS Integration, Scene Description (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (8): AlertConfig, CameraConfig, DepthConfig, DetectorConfig, NetraVisionAiConfig, PipelineConfig, NetraVisionAi — Central Configuration All tuneable parameters in one place, TTSConfig

### Community 16 - "Community 16"
Cohesion: 0.31
Nodes (8): download_file(), download_midas(), export_yolo(), main(), NetraVisionAi — Download all required AI models Run this ONCE before first use, Download a file with progress indicator, Export YOLOv8-nano to ONNX format, Download MiDaS depth estimation model

### Community 17 - "Community 17"
Cohesion: 0.29
Nodes (6): Seq2SeqTrainingArguments, DataTrainingArguments, ModelArguments, ParlerTTSTrainingArguments, Arguments pertaining to what data we are going to input our model for training a, Arguments pertaining to which model/config/tokenizer we are going to fine-tune f

## Knowledge Gaps
- **170 isolated node(s):** `Main NetraVisionAi pipeline.     Captures camera → detects objects → estimates`, `Start the full pipeline.`, `FAST LANE — Main processing loop.         Runs detection + depth on every Nth f`, `Video display loop (runs on main thread).`, `Draw debug visualization on frame.` (+165 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NetraVisionAi` connect `Main Entry` to `Alert Management`, `Depth Estimation`, `TTS Engine`, `Object Detection`, `Spatial Reasoning`, `Parler-TTS Architecture`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `TTSEngine` connect `TTS Engine` to `Main Entry`, `Alert Management`, `Object Detection`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `AlertManager` connect `Alert Management` to `Main Entry`, `TTS Engine`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `ParlerTTSForConditionalGeneration` (e.g. with `ParlerTTSConfig` and `ParlerTTSDecoderConfig`) actually correct?**
  _`ParlerTTSForConditionalGeneration` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `AlertManager` (e.g. with `NetraVisionAi` and `SpeechPriority`) actually correct?**
  _`AlertManager` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `NetraVisionAi` (e.g. with `Camera` and `ObjectDetector`) actually correct?**
  _`NetraVisionAi` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `TTSEngine` (e.g. with `NetraVisionAi` and `AlertManager`) actually correct?**
  _`TTSEngine` has 5 INFERRED edges - model-reasoned connections that need verification._