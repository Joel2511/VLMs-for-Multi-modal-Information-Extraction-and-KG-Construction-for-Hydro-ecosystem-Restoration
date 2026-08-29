#!/usr/bin/env bash

# =============================================================================
# One-document end-to-end demonstration
#
# Selected configuration:
#   Pipeline 1: DocLayout-YOLO extraction
#   Pipeline 2: VDR 2B Multi retrieval
#   Pipeline 3: NVILA-Lite-8B, A5-P3 Counterfactual Contrast
#   Pipeline 4: Fine-tuned Qwen2.5-14B knowledge-graph extraction
#
# Usage:
#   bash scripts/run_demo.sh <pdf-filename-or-stem>
#
# Example:
#   bash scripts/run_demo.sh fichier
#
# This script creates fresh outputs under outputs/runs/demo/<document-name>/.
# It does not overwrite archived thesis outputs.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/config/project.env" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/config/project.env"
fi

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-${PROJECT_ROOT}}"
export HF_MODELS_DIR="${HF_MODELS_DIR:-${RHINE_PROJECT_ROOT}/hf_models}"
export CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/run_demo.sh <pdf-filename-or-stem>"
    exit 1
fi

DOCUMENT_INPUT="$1"
DOCUMENT_STEM="$(basename "${DOCUMENT_INPUT}")"
DOCUMENT_STEM="${DOCUMENT_STEM%.pdf}"

RUN_ROOT="${RHINE_PROJECT_ROOT}/outputs/runs/demo/${DOCUMENT_STEM}"

EXTRACTION_ROOT="${RUN_ROOT}/extraction"
IMAGE_ROOT="${EXTRACTION_ROOT}/extracted_images"
JSON_ROOT="${EXTRACTION_ROOT}/json_results"

RETRIEVAL_ROOT="${RUN_ROOT}/retrieval/vidore_results"
SUMMARY_ROOT="${RUN_ROOT}/summarized_pdfs"
WEIGHTED_CONTEXT_ROOT="${RUN_ROOT}/weighted_context"
CAPTION_ROOT="${RUN_ROOT}/captioning/nvila_p3_contrast_captions"
KG_ROOT="${RUN_ROOT}/knowledge_graph/fine_tuned"
GEXF_ROOT="${RUN_ROOT}/knowledge_graph/gexf"
HTML_ROOT="${RUN_ROOT}/knowledge_graph/html"

LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialization script not found: ${CONDA_SH}"
    echo "Set CONDA_SH in config/project.env."
    exit 1
fi

source "${CONDA_SH}"

run_stage() {
    local stage_name="$1"
    shift

    echo
    echo "======================================================================"
    echo "RUNNING: ${stage_name}"
    echo "======================================================================"

    "$@" 2>&1 | tee "${LOG_ROOT}/${stage_name}.txt"
}

echo "======================================================================"
echo "Rhine River Restoration: One-Document Demonstration"
echo "Document: ${DOCUMENT_INPUT}"
echo "Project root: ${RHINE_PROJECT_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Started: $(date)"
echo "======================================================================"

run_stage "01_extraction" \
    conda run -n kg_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_01_extraction/extract_document_content.py" \
    --document "${DOCUMENT_INPUT}" \
    --pdf-root "${RHINE_PROJECT_ROOT}/data/raw_pdfs" \
    --image-root "${IMAGE_ROOT}" \
    --json-root "${JSON_ROOT}"

run_stage "02_vdr_retrieval" \
    conda run -n barthez_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_02_retrieval/run_vdr.py" \
    --document "${DOCUMENT_STEM}" \
    --image-root "${IMAGE_ROOT}" \
    --json-root "${JSON_ROOT}" \
    --output-root "${RETRIEVAL_ROOT}"

run_stage "03_document_summary" \
    conda run -n vlm_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_03_captioning/context_preparation/summarize_documents.py" \
    --document "${DOCUMENT_STEM}" \
    --input-root "${JSON_ROOT}" \
    --output-root "${SUMMARY_ROOT}"

run_stage "04_weighted_context" \
    conda run -n vlm_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_03_captioning/context_preparation/build_weighted_context.py" \
    --document "${DOCUMENT_STEM}" \
    --json-root "${JSON_ROOT}" \
    --summary-root "${SUMMARY_ROOT}" \
    --retrieval-root "${RUN_ROOT}/retrieval" \
    --output-root "${WEIGHTED_CONTEXT_ROOT}"

run_stage "05_nvila_p3_captioning" \
    conda run -n nvila_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p3_counterfactual_contrast.py" \
    --document "${DOCUMENT_STEM}" \
    --image-root "${IMAGE_ROOT}" \
    --json-root "${JSON_ROOT}" \
    --retrieval-root "${RUN_ROOT}/retrieval" \
    --weighted-context-root "${WEIGHTED_CONTEXT_ROOT}" \
    --output-root "${CAPTION_ROOT}"

run_stage "06_fine_tuned_kg" \
    conda run -n qwen3vl_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_04_knowledge_graph/extract_triples_finetuned.py" \
    --document "${DOCUMENT_STEM}" \
    --captions-root "${CAPTION_ROOT}" \
    --text-root "${JSON_ROOT}" \
    --output-root "${KG_ROOT}"

run_stage "07_ttl_to_gexf" \
    conda run -n qwen3vl_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_04_knowledge_graph/convert_ttl_to_gexf.py" \
    --mode fine_tuned \
    --document "${DOCUMENT_STEM}" \
    --input-root "${KG_ROOT}" \
    --output-root "${GEXF_ROOT}"

run_stage "08_interactive_html" \
    conda run -n qwen3vl_env python \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_04_knowledge_graph/create_interactive_graph.py" \
    "${GEXF_ROOT}/fine_tuned/${DOCUMENT_STEM}/ft_merged_knowledge_graph.gexf" \
    "${HTML_ROOT}/${DOCUMENT_STEM}/interactive_graph.html"

echo
echo "======================================================================"
echo "DEMO COMPLETE"
echo "Finished: $(date)"
echo
echo "Final interactive graph:"
echo "  ${HTML_ROOT}/${DOCUMENT_STEM}/interactive_graph.html"
echo
echo "All demo outputs:"
echo "  ${RUN_ROOT}"
echo "======================================================================"
