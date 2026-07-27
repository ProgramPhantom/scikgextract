"""
Polymer Diffusion Knowledge Extraction Pipeline using SciKGExtract.

This module extracts fitted diffusion coefficients (c or D_0) and Flory parameters (v or nu) 
for ONLY polymers found in scientific literature.
"""

import os
import sys
import json
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv, find_dotenv
import nest_asyncio

# Apply nest_asyncio to allow nested event loops in notebook/interactive execution
nest_asyncio.apply()

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Append workspace root to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# SciKGExtract Imports
from scikg_extract.config.process.processConfig import ProcessConfig
from scikg_extract.config.agents.orchestrator import OrchestratorConfig
from scikg_extract.config.agents.workflow import WorkflowConfig
from scikg_extract.agents.orchestrator_agent import orchestrate_extraction_workflow
from scikg_extract.utils.log_handler import LogHandler


# =============================================================================
# 1. Pydantic Data Models & Schema Definition
# =============================================================================

class QuantityValue(BaseModel):
    """Structured representation of a numeric quantity with units."""
    model_config = ConfigDict(extra='forbid')

    numericValue: Optional[float] = Field(None, description="Extracted numerical value (e.g., 1.2e-4, 0.588).")
    unit: Optional[str] = Field(None, description="Measurement unit (e.g., cm^2/s, m^2/s, K, g/mol).")


class PolymerDiffusionProcess(BaseModel):
    """
    Schema for extracting fitted diffusion parameters for polymers from scientific literature.
    """
    model_config = ConfigDict(extra='forbid')

    polymerName: str = Field(
        ...,
        description="Name, chemical name, or standard abbreviation of the polymer found in literature (e.g., Polystyrene (PS), Poly(ethylene oxide) (PEO), Poly(3-hexylthiophene) (P3HT))."
    )
    isPolymer: bool = Field(
        True,
        description="True if the extracted entity is a polymer/macromolecule. Set to False for non-polymeric small molecules or solvents."
    )
    fittedDiffusionCoefficientC: Optional[QuantityValue] = Field(
        None,
        description="Fitted diffusion coefficient pre-factor c (or D_0) obtained from fitting polymer diffusion equations (e.g., D = c * N^(-v) or D = c * M^(-a))."
    )
    floryParameterV: Optional[float] = Field(
        None,
        description="Flory parameter v (nu) or scaling exponent describing polymer solution behavior (e.g., 0.588 for good solvent, 0.5 for theta solvent, 0.333 for poor solvent)."
    )
    solvent: Optional[str] = Field(
        None,
        description="Solvent or matrix medium in which the polymer diffusion was measured or modeled (e.g., toluene, water, THF)."
    )
    temperature: Optional[QuantityValue] = Field(
        None,
        description="Temperature under which measurements or fits were conducted (e.g., 298.15 K, 25 °C)."
    )
    molecularWeight: Optional[QuantityValue] = Field(
        None,
        description="Molecular weight (Mw or Mn) of the polymer sample if specified."
    )
    fittingEquation: Optional[str] = Field(
        None,
        description="Mathematical equation or scaling model used for fitting (e.g., D = c * N^(-v), D = c * M^(-nu))."
    )
    contextSentence: Optional[str] = Field(
        None,
        description="Direct quote or sentence snippet from the text supporting this extraction."
    )


class PolymerDiffusionProcessList(BaseModel):
    """
    Top-level container for extracted polymer diffusion processes.
    Note: The field name 'processes' is required by SciKGExtract Orchestrator.
    """
    processes: List[PolymerDiffusionProcess] = Field(
        ...,
        description="List of extracted polymer diffusion processes and parameters."
    )


# Generate JSON Schema for extraction
POLYMER_DIFFUSION_SCHEMA = PolymerDiffusionProcessList.model_json_schema()


# =============================================================================
# 2. Domain Process Configuration & Expert Examples
# =============================================================================

DEFAULT_PROCESS_NAME = "Polymer Diffusion Parameter Extraction"

DEFAULT_PROCESS_DESCRIPTION = """
Structured knowledge extraction of fitted polymer diffusion parameters from scientific literature.
This process targets the extraction of the fitted diffusion coefficient pre-factor c (or D_0) 
and the Flory scaling exponent parameter v (nu) describing polymer diffusion dynamics 
(such as D = c * N^(-v) or D = c * M^(-a)) exclusively for polymers (synthetic, natural, block copolymers, etc.) 
reported in research publications.
"""

DEFAULT_PROPERTY_CONSTRAINTS = """
1. Polymer Restriction:
- Extract information ONLY for polymers (macromolecules, synthetic polymers, biopolymers, oligomers).
- Do NOT extract data for non-polymeric small molecules, inorganic salts, or simple solvents unless contextually describing the polymer's solvent environment.
- Explicitly verify that the target entity is a polymer (setting isPolymer=True).

2. Fitted Diffusion Parameters:
- Extract the fitted pre-factor diffusion coefficient c (or D_0 / c) with its numeric value and unit if specified.
- Extract the Flory parameter v (nu) or scaling exponent (typically ~0.588 for good solvents, 0.5 for theta solvents, and 0.333 for poor solvents).

3. Experimental Context:
- Capture the solvent matrix, temperature, polymer molecular weight, and fitting equation whenever stated in the document text.
"""

DEFAULT_EXAMPLES = """
Example 1:
Input Text: "For linear polystyrene (PS) in dilute toluene solution at 25 °C, the self-diffusion coefficient was fitted to the scaling law D = c * M^(-v), yielding a pre-factor c = 1.85e-4 cm^2/s and a Flory exponent v = 0.585 for Mw = 50 kDa."
Extracted JSON:
{
  "processes": [
    {
      "polymerName": "Polystyrene (PS)",
      "isPolymer": true,
      "fittedDiffusionCoefficientC": {
        "numericValue": 0.000185,
        "unit": "cm^2/s"
      },
      "floryParameterV": 0.585,
      "solvent": "toluene",
      "temperature": {
        "numericValue": 25.0,
        "unit": "°C"
      },
      "molecularWeight": {
        "numericValue": 50000.0,
        "unit": "Da"
      },
      "fittingEquation": "D = c * M^(-v)",
      "contextSentence": "For linear polystyrene (PS) in dilute toluene solution at 25 °C, the self-diffusion coefficient was fitted to the scaling law D = c * M^(-v), yielding a pre-factor c = 1.85e-4 cm^2/s and a Flory exponent v = 0.585 for Mw = 50 kDa."
    }
  ]
}

Example 2:
Input Text: "We investigated Poly(ethylene oxide) (PEO) in aqueous solution at 298 K. The fitted diffusion coefficient was c = 2.1e-4 cm^2/s with a Flory parameter v = 0.588. Small molecule glucose was also measured with D = 6.7e-6 cm^2/s."
Extracted JSON:
{
  "processes": [
    {
      "polymerName": "Poly(ethylene oxide) (PEO)",
      "isPolymer": true,
      "fittedDiffusionCoefficientC": {
        "numericValue": 0.00021,
        "unit": "cm^2/s"
      },
      "floryParameterV": 0.588,
      "solvent": "aqueous solution (water)",
      "temperature": {
        "numericValue": 298.0,
        "unit": "K"
      },
      "molecularWeight": null,
      "fittingEquation": null,
      "contextSentence": "We investigated Poly(ethylene oxide) (PEO) in aqueous solution at 298 K. The fitted diffusion coefficient was c = 2.1e-4 cm^2/s with a Flory parameter v = 0.588."
    }
  ]
}
"""


# =============================================================================
# 3. Main Extraction Pipeline Function
# =============================================================================

def extract_polymer_diffusion_parameters(
    scientific_document: str,
    extraction_llm: str = "OLLAMA:gemma4:31b",
    normalize_data: bool = False,
    validate_data: bool = False,
    refine_data: bool = False,
    examples: str = DEFAULT_EXAMPLES,
) -> dict:
    """
    Executes the SciKGExtract pipeline to extract fitted diffusion coefficient 'c' 
    and Flory parameter 'v' for polymers found in the input scientific document.

    Args:
        scientific_document (str): Markdown or plain text content of the scientific paper.
        extraction_llm (str): Model string for extraction (default: "OLLAMA:gemma4:31b").
        normalize_data (bool): Whether to perform PubChem entity normalization (default: False).
        validate_data (bool): Whether to perform LLM-as-a-Judge reflection (default: False).
        refine_data (bool): Whether to iteratively refine extraction based on feedback (default: False).
        examples (str): Domain expert examples string.

    Returns:
        dict: Extracted JSON dictionary conforming to PolymerDiffusionProcessList schema.
    """
    # Setup logger
    logger = LogHandler.setup_module_logging("scikg_extract")
    logger.info("Initializing Polymer Diffusion Parameter Extraction Pipeline...")

    # Configure ProcessConfig for polymer diffusion domain
    ProcessConfig.Process_name = DEFAULT_PROCESS_NAME
    ProcessConfig.Process_description = DEFAULT_PROCESS_DESCRIPTION
    ProcessConfig.Process_property_constraints = DEFAULT_PROPERTY_CONSTRAINTS

    # Build Orchestrator Agent Configuration
    orchestrator_config = OrchestratorConfig(
        extraction_llm=extraction_llm,
        process_schema=POLYMER_DIFFUSION_SCHEMA,
        scientific_document=scientific_document,
        examples=examples,
        extraction_data_model=PolymerDiffusionProcessList
    )

    # Build Workflow Configuration
    workflow_config = WorkflowConfig(
        normalize_extracted_data=normalize_data,
        clean_extracted_data=True,
        validate_extracted_data=validate_data,
        refine_extracted_data=refine_data
    )

    # Run extraction workflow
    logger.info(f"Running extraction workflow with LLM: {extraction_llm}...")
    final_state = orchestrate_extraction_workflow(orchestrator_config, workflow_config)

    extracted_knowledge = final_state.get("extracted_json", {})
    return extracted_knowledge


# =============================================================================
# 4. CLI / Demonstration Entrypoint
# =============================================================================

if __name__ == "__main__":
    # Sample scientific document snippet for demonstration
    sample_paper = """
    # Polymer Diffusion Dynamics in Solution
    
    ## Abstract
    We measure the translational diffusion of linear polymers in various solvent conditions using pulsed-field gradient NMR spectroscopy.
    
    ## Results and Discussion
    The self-diffusion of Polystyrene (PS) in dilute toluene at T = 298.15 K was measured across a range of molecular weights (Mw from 10 kDa to 500 kDa).
    Fitting the molecular weight dependence to the scaling relationship D = c * Mw^(-v) yielded a fitted diffusion coefficient pre-factor c = 1.45e-4 cm^2/s 
    and a Flory parameter v = 0.588, consistent with good solvent statistics.
    
    In comparison, Poly(methyl methacrylate) (PMMA) dissolved in acetone at 25 °C exhibited a fitted diffusion coefficient c = 2.05e-4 cm^2/s 
    and Flory parameter v = 0.500 under theta conditions.
    
    Additionally, diffusion of the small-molecule additive toluene was recorded as D = 2.2e-5 cm^2/s; however, toluene is a low-molecular-weight solvent.
    """

    # Check for OLLAMA model setting or environment override
    llm_model = os.getenv("LLM_MODEL", "OLLAMA:gemma4:31b")
    
    print("=" * 80)
    print("Polymer Diffusion Parameter Extraction Pipeline")
    print(f"Using LLM: {llm_model}")
    print("=" * 80)
    
    try:
        results = extract_polymer_diffusion_parameters(
            scientific_document=sample_paper,
            extraction_llm=llm_model
        )
        print("\nExtracted Knowledge Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\nPipeline execution encounter: {e}")
        print("\nEnsure Ollama is running locally with gemma4:31b (`ollama run gemma4:31b`), or set LLM_MODEL env variable.")
