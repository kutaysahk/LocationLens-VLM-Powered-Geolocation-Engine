# ◈ LocationLens
VLM-Powered Geolocation Engine

LocationLens is an intelligent scene-recognition and geolocation tool that identifies physical locations from images using a multi-model approach. It combines the visual reasoning of Qwen2.5-VL with a custom Spatial Intersection Engine to cross-reference identified signage against OpenStreetMap (OSM) data.

Key Features

VLM Sign Extraction: Uses 4-bit quantized Qwen2.5-VL-3B to read business names, shop signs, and landmarks with per-sign confidence weighting.

Intersection Engine: A robust algorithm that tries every detected sign as a spatial anchor, looking for clusters of businesses within a 200m radius to verify location accuracy.

EXIF Intelligence: Automatically prioritizes embedded GPS data for ground-truth accuracy while using the VLM for semantic scene categorization.

Hybrid Search: Implements a multi-strategy parser to rescue text from truncated or malformed JSON outputs.

Cyberpunk UI: A responsive React frontend featuring a "scanline" animation, real-time status updates, and a comprehensive scan history.